#!/usr/bin/env python3
"""
Minimal Gluon GEMM test - directly use gl.load/store and gl.amd.cdna4.mfma
without explicit shared memory management.
"""

import torch
import triton
from triton import language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna4
import time

@gluon.jit
def matmul_simple_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr, NUM_WARPS: gl.constexpr,
):
    """Minimal Gluon GEMM - let compiler handle shared memory."""
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    
    # Swizzle for L2 cache
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Use coalesced layout - let Triton pick optimal
    layout_a: gl.constexpr = gl.CoalescedLayout()
    layout_b: gl.constexpr = gl.CoalescedLayout()
    
    # Create offset arrays
    offs_am = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.AutoLayout())
    offs_bn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.AutoLayout())
    offs_k = gl.arange(0, BLOCK_K, layout=gl.AutoLayout())
    
    # Compute base pointers
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Initialize accumulator
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=gl.AutoLayout())
    
    num_k_iters = gl.cdiv(K, BLOCK_K)
    
    for k in range(num_k_iters):
        k_offset = k * BLOCK_K
        
        # Load tiles
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] + k_offset < K)
        b_mask = (offs_k[:, None] + k_offset < K) & (offs_bn[None, :] < N)
        
        a = gl.load(a_ptrs + k_offset * stride_ak, mask=a_mask, other=0.0)
        b = gl.load(b_ptrs + k_offset * stride_bk, mask=b_mask, other=0.0)
        
        # Dot product
        acc += gl.sum(a[:, :, None] * b[None, :, :], 1)
    
    # Store result
    c = acc.to(gl.bfloat16)
    
    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.AutoLayout())
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.AutoLayout())
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    gl.store(c_ptrs, c, mask=c_mask)


def matmul_simple(a, b, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, NUM_WARPS=4):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    GROUP_SIZE_M = 8
    
    grid = lambda META: (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    matmul_simple_kernel[grid](
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
    print(f"\nTesting simple Gluon GEMM: {N}x{N} @ {N}x{N}")
    
    a = torch.randn(N, N, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(N, N, device='cuda', dtype=torch.bfloat16)
    
    ref = torch.matmul(a, b)
    ref_time = benchmark(lambda: torch.matmul(a, b))
    print(f"rocBLAS time: {ref_time:.3f}ms")
    
    try:
        out = matmul_simple(a, b)
        max_diff = (ref - out).abs().max().item()
        print(f"Max diff: {max_diff}")
        
        if max_diff > 1.0:
            print("ERROR: Accuracy check failed!")
        else:
            gluon_time = benchmark(lambda: matmul_simple(a, b))
            print(f"Simple Gluon time: {gluon_time:.3f}ms ({ref_time/gluon_time:.3f}x)")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

