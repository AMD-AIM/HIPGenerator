#!/usr/bin/env python3
"""Extract Triton-generated assembly for analysis."""

import torch
import triton
import triton.language as tl
import os
import re

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, 
                     num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def main():
    M, N, K = 4096, 4096, 4096
    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    c = torch.empty((M, N), device='cuda', dtype=torch.bfloat16)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    # Run to compile
    matmul_kernel[grid](a, b, c, M, N, K, 
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    torch.cuda.synchronize()
    
    # Find the cached kernel
    cache_dir = os.environ.get('TRITON_CACHE_DIR', os.path.expanduser('~/.triton/cache'))
    print(f"Looking in cache dir: {cache_dir}")
    
    # Find ASM files
    asm_files = []
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith('.amdgcn') or 'amdgcn' in f:
                asm_files.append(os.path.join(root, f))
    
    if asm_files:
        # Get the most recent one
        asm_files.sort(key=os.path.getmtime, reverse=True)
        asm_path = asm_files[0]
        print(f"Found ASM: {asm_path}")
        
        with open(asm_path, 'r') as f:
            asm = f.read()
    else:
        # Try to get from kernel object
        print("No ASM files found, trying kernel introspection...")
        
        # Check kernel cache
        for key, compiled in matmul_kernel.cache.items():
            print(f"Cache key: {key}")
            if hasattr(compiled, 'asm'):
                print(f"  Has asm: {list(compiled.asm.keys()) if compiled.asm else 'empty'}")
                if 'amdgcn' in compiled.asm:
                    asm = compiled.asm['amdgcn']
                    break
        else:
            print("Could not find ASM")
            return
    
    # Save and analyze
    with open('triton_bf16_gemm.s', 'w') as f:
        f.write(asm)
    print(f"\nSaved Triton assembly ({len(asm)} bytes)")
    
    # Count instructions
    mfma = len(re.findall(r'v_mfma', asm))
    mfma_types = re.findall(r'v_mfma_[a-z0-9_]+', asm)
    global_load = len(re.findall(r'global_load', asm))
    buffer_load = len(re.findall(r'buffer_load', asm))
    ds_read = len(re.findall(r'ds_read', asm))
    ds_write = len(re.findall(r'ds_write', asm))
    barrier = len(re.findall(r's_barrier', asm))
    waitcnt = len(re.findall(r's_waitcnt', asm))
    setprio = len(re.findall(r's_setprio', asm))
    
    print(f"\n=== Triton Instruction Analysis ===")
    print(f"  MFMA:         {mfma}")
    print(f"  Global load:  {global_load}")
    print(f"  Buffer load:  {buffer_load}")
    print(f"  LDS read:     {ds_read}")
    print(f"  LDS write:    {ds_write}")
    print(f"  Barrier:      {barrier}")
    print(f"  Waitcnt:      {waitcnt}")
    print(f"  s_setprio:    {setprio}")
    
    if mfma_types:
        print(f"\nMFMA types:")
        type_counts = {}
        for t in mfma_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")
    
    # Show a sample of the MFMA region
    print("\n=== Sample MFMA region ===")
    lines = asm.split('\n')
    for i, line in enumerate(lines):
        if 'v_mfma' in line:
            start = max(0, i-5)
            end = min(len(lines), i+30)
            for j in range(start, end):
                marker = ">>>" if 'v_mfma' in lines[j] else "   "
                print(f"{marker} {lines[j]}")
            break


if __name__ == "__main__":
    main()


