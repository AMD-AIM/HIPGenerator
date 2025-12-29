#!/usr/bin/env python3
"""
Analyze Triton assembly with and without pingpong scheduling to see the differences.
"""

import os
import torch
import triton
import triton.language as tl
import tempfile
import subprocess
import sys

# First create a kernel file that can be imported
KERNEL_FILE = "/tmp/triton_matmul_kernel.py"

kernel_source = '''
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask)
'''

def analyze_asm(asm, name):
    """Analyze assembly for key patterns"""
    lines = asm.split('\n')
    
    mfma_count = sum(1 for l in lines if 'v_mfma' in l)
    setprio_count = sum(1 for l in lines if 's_setprio' in l)
    sched_barrier = sum(1 for l in lines if 's_sched_barrier' in l)
    lgkmcnt_0 = sum(1 for l in lines if 'lgkmcnt(0)' in l)
    lgkmcnt_partial = sum(1 for l in lines if 'lgkmcnt(' in l and 'lgkmcnt(0)' not in l)
    vmcnt_0 = sum(1 for l in lines if 'vmcnt(0)' in l)
    vmcnt_partial = sum(1 for l in lines if 'vmcnt(' in l and 'vmcnt(0)' not in l)
    ds_read = sum(1 for l in lines if 'ds_read' in l)
    ds_write = sum(1 for l in lines if 'ds_write' in l)
    global_load = sum(1 for l in lines if 'global_load' in l or 'buffer_load' in l)
    s_barrier = sum(1 for l in lines if 's_barrier' in l and 's_sched_barrier' not in l)
    
    print(f"\n=== {name} ===")
    print(f"  MFMA instructions: {mfma_count}")
    print(f"  s_setprio: {setprio_count}")
    print(f"  s_sched_barrier: {sched_barrier}")
    print(f"  s_barrier: {s_barrier}")
    print(f"  lgkmcnt(0): {lgkmcnt_0}, lgkmcnt(N>0): {lgkmcnt_partial}")
    print(f"  vmcnt(0): {vmcnt_0}, vmcnt(N>0): {vmcnt_partial}")
    print(f"  ds_read: {ds_read}, ds_write: {ds_write}")
    print(f"  global_load: {global_load}")
    
    # Show setprio context
    if setprio_count > 0:
        print(f"\n  s_setprio context:")
        count = 0
        for i, line in enumerate(lines):
            if 's_setprio' in line and count < 3:
                start = max(0, i-2)
                end = min(len(lines), i+3)
                for j in range(start, end):
                    prefix = ">>>" if j == i else "   "
                    print(f"    {prefix} {lines[j]}")
                print()
                count += 1


def run_and_extract_asm(use_pingpong, output_file):
    """Run in subprocess to get fresh compilation with env vars"""
    
    # Write kernel file
    with open(KERNEL_FILE, 'w') as f:
        f.write(kernel_source)
    
    script = f'''
import os
import sys
import tempfile

# Set cache dir
cache_dir = tempfile.mkdtemp()
os.environ["TRITON_CACHE_DIR"] = cache_dir
os.environ["AMDGCN_ENABLE_DUMP"] = "1"

# Set pingpong
if {use_pingpong}:
    os.environ["TRITON_HIP_USE_BLOCK_PINGPONG"] = "1"

import torch
import triton

# Import kernel from file
sys.path.insert(0, "/tmp")
from triton_matmul_kernel import matmul_kernel

M, N, K = 4096, 4096, 4096

a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 32
grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

matmul_kernel[grid](
    a, b, c, M, N, K,
    a.stride(0), a.stride(1),
    b.stride(0), b.stride(1),
    c.stride(0), c.stride(1),
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    GROUP_M=8,
    num_warps=8, num_stages=2,
)

# Find the amdgcn file
import os
for root, dirs, files in os.walk(cache_dir):
    for f in files:
        if f.endswith(".amdgcn"):
            asm_path = os.path.join(root, f)
            with open(asm_path) as af:
                print(af.read())
            break
'''
    
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error: {result.stderr[:1000]}")
        return None
    
    # Save and return assembly
    with open(output_file, "w") as f:
        f.write(result.stdout)
    
    return result.stdout


def main():
    print("Analyzing Triton assembly with and without pingpong scheduling")
    print("=" * 80)
    
    # Get assembly without pingpong
    print("\nCompiling baseline (no pingpong)...")
    asm_baseline = run_and_extract_asm(False, "triton_baseline.s")
    if asm_baseline:
        analyze_asm(asm_baseline, "Baseline (no pingpong)")
    
    # Get assembly with pingpong
    print("\nCompiling with pingpong...")
    asm_pingpong = run_and_extract_asm(True, "triton_pingpong.s")
    if asm_pingpong:
        analyze_asm(asm_pingpong, "With Pingpong")
    
    print("\n" + "=" * 80)
    print("Assembly saved to triton_baseline.s and triton_pingpong.s")


if __name__ == "__main__":
    main()
