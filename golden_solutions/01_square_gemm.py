#!/usr/bin/env python3
"""
Problem: 1_Square_matrix_multiplication_
Shape: (4096, 4096) x (4096, 4096) -> (4096, 4096)
Target: MI350 (gfx950), 32 XCDs, 256 CUs

Optimizations:
- XCD Swizzle for 32 XCDs
- Pingpong scheduling (TRITON_HIP_USE_BLOCK_PINGPONG=1)
- 16x16 MFMA (matrix_instr_nonkdim=16)
- Aligned fast path (no masking)
- 256x256x32 tiles, 3 stages, 8 warps
- L2 cache grouping (GROUP_M=8)
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import os

os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'

NUM_XCDS = 32

# ============ Original Model (Reference) ============
class Model(nn.Module):
    """Original model that performs a single square matrix multiplication (C = A * B)"""
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)


# ============ Optimized Triton Kernel ============
@triton.jit
def matmul_kernel_aligned(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_XCDS: tl.constexpr,
):
    """Optimized kernel for large aligned matrices with XCD swizzle."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pids = num_pid_m * num_pid_n
    
    # XCD-aware swizzle for 32 XCDs
    pids_per_xcd = (num_pids + NUM_XCDS - 1) // NUM_XCDS
    xcd_id = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    
    if local_pid < pids_per_xcd:
        remapped_pid = xcd_id * pids_per_xcd + local_pid
        if remapped_pid < num_pids:
            pid = remapped_pid
    
    # L2 cache-friendly grouping
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # No masking for aligned matrices
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
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Optimal config for 4096x4096 on MI350
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 32
    num_stages, num_warps = 3, 8
    GROUP_M = 8
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    matmul_kernel_aligned[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M, NUM_XCDS=NUM_XCDS,
        num_stages=num_stages, num_warps=num_warps, matrix_instr_nonkdim=16,
    )
    return c


class ModelNew(nn.Module):
    """Optimized model using Triton kernel."""
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


# ============ Test Inputs ============
N = 2048 * 2  # 4096

def get_inputs():
    A = torch.rand(N, N, dtype=torch.float16)
    B = torch.rand(N, N, dtype=torch.float16)
    return [A, B]

def get_init_inputs():
    return []


# ============ Verification ============
if __name__ == "__main__":
    A, B = get_inputs()
    A, B = A.cuda(), B.cuda()
    
    ref_model = Model().cuda()
    new_model = ModelNew().cuda()
    
    ref = ref_model(A, B)
    out = new_model(A, B)
    
    max_diff = (ref.float() - out.float()).abs().max().item()
    print(f"Max diff: {max_diff}")
    
    # Benchmark
    import time
    torch.cuda.synchronize()
    
    for _ in range(10):
        _ = new_model(A, B)
    torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(100):
        _ = new_model(A, B)
    torch.cuda.synchronize()
    triton_time = (time.time() - t0) / 100
    
    for _ in range(10):
        _ = ref_model(A, B)
    torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(100):
        _ = ref_model(A, B)
    torch.cuda.synchronize()
    ref_time = (time.time() - t0) / 100
    
    speedup = ref_time / triton_time
    tflops = 2 * N * N * N / triton_time / 1e12
    
    print(f"Triton: {triton_time*1000:.3f}ms, {tflops:.2f} TFLOPS")
    print(f"rocBLAS: {ref_time*1000:.3f}ms")
    print(f"Speedup: {speedup:.3f}x")
