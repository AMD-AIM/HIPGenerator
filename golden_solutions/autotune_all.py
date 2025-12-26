#!/usr/bin/env python3
"""
Systematic autotuning for all golden solutions.
Tests multiple configurations and finds the best one for each case.
"""

import torch
import triton
import triton.language as tl
import time
import os
import sys
from typing import List, Tuple, Dict, Any

os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'

NUM_XCDS = 32

# ============ Generic GEMM Kernel for Testing ============
@triton.jit
def matmul_kernel_tunable(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_XCDS: tl.constexpr,
):
    """Generic GEMM kernel for autotuning."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pids = num_pid_m * num_pid_n
    
    # XCD swizzle
    pids_per_xcd = (num_pids + NUM_XCDS - 1) // NUM_XCDS
    xcd_id = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if local_pid < pids_per_xcd:
        remapped_pid = xcd_id * pids_per_xcd + local_pid
        if remapped_pid < num_pids:
            pid = remapped_pid
    
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))


@triton.jit  
def bmm_kernel_tunable(
    a_ptr, b_ptr, c_ptr, B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Batched GEMM kernel for autotuning."""
    batch_id = tl.program_id(1)
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_batch = a_ptr + batch_id * stride_ab
    b_batch = b_ptr + batch_id * stride_bb
    c_batch = c_ptr + batch_id * stride_cb
    
    a_ptrs = a_batch + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_batch + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = c_batch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))


def test_gemm_config(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, GROUP_M=8):
    """Test a single GEMM configuration."""
    # Check divisibility for unmasked loads
    if M % BLOCK_M != 0 or N % BLOCK_N != 0 or K % BLOCK_K != 0:
        return 0.0
    
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    C = torch.empty(M, N, dtype=torch.float16, device='cuda')
    
    grid = ((M // BLOCK_M) * (N // BLOCK_N),)
    
    try:
        os.system('rm -rf ~/.triton/cache 2>/dev/null')
        
        # Warmup
        for _ in range(5):
            matmul_kernel_tunable[grid](
                A, B, C, M, N, K,
                A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                GROUP_M=GROUP_M, NUM_XCDS=NUM_XCDS,
                num_stages=num_stages, num_warps=num_warps, matrix_instr_nonkdim=16,
            )
        torch.cuda.synchronize()
        
        # Benchmark
        t0 = time.time()
        for _ in range(50):
            matmul_kernel_tunable[grid](
                A, B, C, M, N, K,
                A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                GROUP_M=GROUP_M, NUM_XCDS=NUM_XCDS,
                num_stages=num_stages, num_warps=num_warps, matrix_instr_nonkdim=16,
            )
        torch.cuda.synchronize()
        triton_time = (time.time() - t0) / 50
        
        # Reference
        for _ in range(5):
            _ = torch.matmul(A, B)
        torch.cuda.synchronize()
        
        t0 = time.time()
        for _ in range(50):
            _ = torch.matmul(A, B)
        torch.cuda.synchronize()
        ref_time = (time.time() - t0) / 50
        
        return ref_time / triton_time
    except Exception as e:
        return 0.0


def test_bmm_config(Batch, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, GROUP_M=8):
    """Test a single BMM configuration."""
    if M % BLOCK_M != 0 or N % BLOCK_N != 0 or K % BLOCK_K != 0:
        return 0.0
    
    A = torch.randn(Batch, M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(Batch, K, N, dtype=torch.float16, device='cuda')
    C = torch.empty(Batch, M, N, dtype=torch.float16, device='cuda')
    
    grid = ((M // BLOCK_M) * (N // BLOCK_N), Batch)
    
    try:
        os.system('rm -rf ~/.triton/cache 2>/dev/null')
        
        for _ in range(5):
            bmm_kernel_tunable[grid](
                A, B, C, Batch, M, N, K,
                A.stride(0), A.stride(1), A.stride(2),
                B.stride(0), B.stride(1), B.stride(2),
                C.stride(0), C.stride(1), C.stride(2),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
                num_stages=num_stages, num_warps=num_warps, matrix_instr_nonkdim=16,
            )
        torch.cuda.synchronize()
        
        t0 = time.time()
        for _ in range(30):
            bmm_kernel_tunable[grid](
                A, B, C, Batch, M, N, K,
                A.stride(0), A.stride(1), A.stride(2),
                B.stride(0), B.stride(1), B.stride(2),
                C.stride(0), C.stride(1), C.stride(2),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
                num_stages=num_stages, num_warps=num_warps, matrix_instr_nonkdim=16,
            )
        torch.cuda.synchronize()
        triton_time = (time.time() - t0) / 30
        
        for _ in range(5):
            _ = torch.bmm(A, B)
        torch.cuda.synchronize()
        
        t0 = time.time()
        for _ in range(30):
            _ = torch.bmm(A, B)
        torch.cuda.synchronize()
        ref_time = (time.time() - t0) / 30
        
        return ref_time / triton_time
    except Exception as e:
        return 0.0


def generate_configs():
    """Generate all possible configurations to test."""
    configs = []
    
    for BLOCK_M in [64, 128, 256]:
        for BLOCK_N in [64, 128, 256]:
            for BLOCK_K in [32, 64]:
                for num_stages in [2, 3]:
                    for num_warps in [4, 8]:
                        for GROUP_M in [4, 8]:
                            configs.append((BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, GROUP_M))
    
    return configs


def autotune_case(name: str, test_fn, *args) -> Tuple[Dict, float]:
    """Autotune a single case."""
    print(f"\n{'='*60}")
    print(f"Autotuning: {name}")
    print(f"{'='*60}")
    
    configs = generate_configs()
    best_speedup = 0.0
    best_config = None
    
    for i, (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, GROUP_M) in enumerate(configs):
        speedup = test_fn(*args, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, GROUP_M)
        if speedup > 0:
            marker = "⭐" if speedup > best_speedup else ""
            if speedup > best_speedup:
                best_speedup = speedup
                best_config = {
                    'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'BLOCK_K': BLOCK_K,
                    'num_stages': num_stages, 'num_warps': num_warps, 'GROUP_M': GROUP_M
                }
                print(f"[{i+1}/{len(configs)}] BLOCK=({BLOCK_M},{BLOCK_N},{BLOCK_K}), stages={num_stages}, warps={num_warps}, GROUP_M={GROUP_M}: {speedup:.3f}x {marker}")
    
    print(f"\nBest config for {name}: {best_config}")
    print(f"Best speedup: {best_speedup:.3f}x")
    
    return best_config, best_speedup


def main():
    print("="*60)
    print("TRITON AUTOTUNE FOR ALL GOLDEN SOLUTIONS")
    print("="*60)
    
    results = {}
    
    # Case 01: Square GEMM (4096x4096)
    config, speedup = autotune_case(
        "01_square_gemm (4096x4096)", 
        test_gemm_config, 4096, 4096, 4096
    )
    results['01_square_gemm'] = {'config': config, 'speedup': speedup}
    
    # Case 02: Batched GEMM (128x512x2048x1024)
    config, speedup = autotune_case(
        "02_batched_gemm (128x512x2048x1024)",
        test_bmm_config, 128, 512, 2048, 1024
    )
    results['02_batched_gemm'] = {'config': config, 'speedup': speedup}
    
    # Case 04: GEMM for bias+relu (1024x8192)
    config, speedup = autotune_case(
        "04_gemm_bias_relu (1024x8192x8192)",
        test_gemm_config, 1024, 8192, 8192
    )
    results['04_gemm_bias_relu'] = {'config': config, 'speedup': speedup}
    
    # Case 05: GEMM for divide+gelu (1024x8192)
    config, speedup = autotune_case(
        "05_gemm_divide_gelu (1024x8192x8192)",
        test_gemm_config, 1024, 8192, 8192
    )
    results['05_gemm_divide_gelu'] = {'config': config, 'speedup': speedup}
    
    # Case 06: Tall skinny (16384x1024x16)
    config, speedup = autotune_case(
        "06_tall_skinny (16384x1024x16)",
        test_gemm_config, 16384, 1024, 16
    )
    results['06_tall_skinny'] = {'config': config, 'speedup': speedup}
    
    # Case 07: GEMM for swish+scaling (1024x4096x4096)
    config, speedup = autotune_case(
        "07_gemm_swish_scaling (1024x4096x4096)",
        test_gemm_config, 1024, 4096, 4096
    )
    results['07_gemm_swish_scaling'] = {'config': config, 'speedup': speedup}
    
    # Case 08: Rectangular GEMM (1024x2048x4096)
    config, speedup = autotune_case(
        "08_rectangular_gemm (1024x2048x4096)",
        test_gemm_config, 1024, 2048, 4096
    )
    results['08_rectangular_gemm'] = {'config': config, 'speedup': speedup}
    
    # Case 09: GEMM for sigmoid+sum (1024x4096x4096)
    config, speedup = autotune_case(
        "09_gemm_sigmoid_sum (1024x4096x4096)",
        test_gemm_config, 1024, 4096, 4096
    )
    results['09_gemm_sigmoid_sum'] = {'config': config, 'speedup': speedup}
    
    # Case 10: GEMM for gelu+softmax (1024x4096x4096)
    config, speedup = autotune_case(
        "10_gemm_gelu_softmax (1024x4096x4096)",
        test_gemm_config, 1024, 4096, 4096
    )
    results['10_gemm_gelu_softmax'] = {'config': config, 'speedup': speedup}
    
    # Print summary
    print("\n" + "="*60)
    print("AUTOTUNE SUMMARY")
    print("="*60)
    print(f"\n{'Case':<30} {'Best Speedup':<15} {'Best Config'}")
    print("-"*80)
    for case, data in results.items():
        cfg = data['config']
        if cfg:
            cfg_str = f"({cfg['BLOCK_M']},{cfg['BLOCK_N']},{cfg['BLOCK_K']}), s={cfg['num_stages']}, w={cfg['num_warps']}"
        else:
            cfg_str = "N/A"
        print(f"{case:<30} {data['speedup']:.3f}x          {cfg_str}")
    
    # Save results
    import json
    with open('autotune_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to autotune_results.json")


if __name__ == "__main__":
    main()

