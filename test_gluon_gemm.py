#!/usr/bin/env python3
"""
Triton Gluon GEMM for AMD MI300X/MI350X.

Uses cdna4.buffer_load + cdna4.mfma (no async_copy due to compilation issues).
"""

import torch
import triton
from triton import language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna4
import time

@gluon.jit
def matmul_gluon_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr, NUM_WARPS: gl.constexpr,
):
    """Gluon GEMM with explicit buffer_load and MFMA."""
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    
    # Grouped ordering for L2 cache
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Layout definitions
    threads_per_elem_mk: gl.constexpr = triton.cdiv(BLOCK_M * BLOCK_K // (NUM_WARPS * 64), 16)
    threads_per_elem_kn: gl.constexpr = triton.cdiv(BLOCK_K * BLOCK_N // (NUM_WARPS * 64), 16)
    
    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[threads_per_elem_mk, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, threads_per_elem_kn],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )
    
    # MFMA layout
    warps_m: gl.constexpr = NUM_WARPS // 2
    warps_n: gl.constexpr = 2
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16],
        transposed=True,
        warps_per_cta=[warps_m, warps_n],
    )
    
    # Shared memory layouts with optimal swizzling
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0, 1]
    )
    
    # Dot operand layouts - use k_width=8 for bf16
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )
    
    # Allocate shared memory
    smem_a = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, BLOCK_K], layout=shared_a)
    smem_b = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_K, BLOCK_N], layout=shared_b)
    
    # Initialize accumulator in MFMA layout
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout)
    
    # Create offset arrays
    offs_ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blocked_mk))
    offs_bk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, blocked_kn))
    offs_am = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_mk))
    offs_bn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kn))
    
    # Compute base offsets
    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    num_k_iters = gl.cdiv(K, BLOCK_K)
    
    # Main loop
    for k in range(num_k_iters):
        k_offset = k * BLOCK_K
        
        # Compute masks for boundary checking
        a_mask = (offs_am[:, None] < M) & (offs_ak[None, :] + k_offset < K)
        b_mask = (offs_bk[:, None] + k_offset < K) & (offs_bn[None, :] < N)
        
        # Load tiles using buffer_load (goes through registers)
        a = cdna4.buffer_load(ptr=a_ptr, offsets=offs_a + k_offset * stride_ak, mask=a_mask, cache=".ca")
        b = cdna4.buffer_load(ptr=b_ptr, offsets=offs_b + k_offset * stride_bk, mask=b_mask, cache=".ca")
        
        # Store to shared memory
        smem_a.store(a)
        smem_b.store(b)
        
        # Barrier to ensure stores complete
        gl.thread_barrier()
        
        # Load from shared with MFMA-compatible layout
        cur_a = smem_a.load(layout=dot_a_layout)
        cur_b = smem_b.load(layout=dot_b_layout)
        
        # MFMA operation
        acc = cdna4.mfma(cur_a, cur_b, acc)
        
        # Barrier before next iteration
        gl.thread_barrier()
    
    # Store result
    c = acc.to(gl.bfloat16)
    
    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
    c_offs = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    cdna4.buffer_store(stored_value=c, ptr=c_ptr, offsets=c_offs, mask=c_mask)


def matmul_gluon(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, NUM_WARPS=4):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    GROUP_SIZE_M = 8
    
    grid = lambda META: (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    matmul_gluon_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M, NUM_WARPS=NUM_WARPS,
    )
    return c


def benchmark(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


if __name__ == "__main__":
    print(f"Triton version: {triton.__version__}")
    
    N = 4096
    print(f"\nTesting Gluon GEMM: {N}x{N} @ {N}x{N}")
    
    a = torch.randn(N, N, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(N, N, device='cuda', dtype=torch.bfloat16)
    
    ref = torch.matmul(a, b)
    ref_time = benchmark(lambda: torch.matmul(a, b))
    print(f"rocBLAS time: {ref_time:.3f}ms")
    
    # Test configurations
    configs = [
        (128, 128, 64, 4),
        (128, 128, 32, 4),
        (64, 64, 64, 4),
        (64, 128, 64, 4),
        (128, 64, 64, 4),
    ]
    
    best_speedup = 0
    best_config = None
    
    for BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS in configs:
        try:
            out = matmul_gluon(a, b, BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS)
            max_diff = (ref - out).abs().max().item()
            
            if max_diff > 1.0:
                print(f"  {BLOCK_M}x{BLOCK_N}x{BLOCK_K} warps={NUM_WARPS}: FAILED (diff={max_diff})")
                continue
                
            gluon_time = benchmark(lambda: matmul_gluon(a, b, BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS))
            speedup = ref_time / gluon_time
            print(f"  {BLOCK_M}x{BLOCK_N}x{BLOCK_K} warps={NUM_WARPS}: {gluon_time:.3f}ms ({speedup:.3f}x)")
            
            if speedup > best_speedup:
                best_speedup = speedup
                best_config = (BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS)
        except Exception as e:
            err_msg = str(e)[:150]
            print(f"  {BLOCK_M}x{BLOCK_N}x{BLOCK_K} warps={NUM_WARPS}: ERROR - {err_msg}")
    
    print(f"\nBest Gluon config: {best_config} with speedup {best_speedup:.3f}x")
    
    # Compare with standard Triton
    print("\n--- Comparison with Standard Triton ---")
    try:
        from test_autotune import matmul as triton_matmul
        triton_out = triton_matmul(a, b)
        triton_time = benchmark(lambda: triton_matmul(a, b))
        print(f"Standard Triton: {triton_time:.3f}ms ({ref_time/triton_time:.3f}x)")
    except Exception as e:
        print(f"Standard Triton: {e}")
