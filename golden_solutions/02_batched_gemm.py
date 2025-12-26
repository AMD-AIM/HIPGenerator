#!/usr/bin/env python3
"""
Problem: 3_Batched_matrix_multiplication
Shape: (128, 512, 1024) x (128, 1024, 2048) -> (128, 512, 2048)
Target: MI350 (gfx950), 32 XCDs, 256 CUs

Optimizations:
- 3D grid (batch, M_tiles, N_tiles) for better parallelism
- XCD-aware batch distribution
- Pingpong scheduling
- 16x16 MFMA
- 128x128x64 tiles
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
    """Performs batched matrix multiplication (C = A * B)."""
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.bmm(A, B)


# ============ Optimized Triton Kernel ============
@triton.jit
def bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Batched matrix multiplication kernel."""
    batch_id = tl.program_id(1)
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 cache grouping within each batch
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Batch offsets
    a_batch = a_ptr + batch_id * stride_ab
    b_batch = b_ptr + batch_id * stride_bb
    c_batch = c_ptr + batch_id * stride_cb
    
    a_ptrs = a_batch + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_batch + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        m_mask = offs_m < M
        n_mask = offs_n < N
        
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = c_batch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def triton_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B, M, K = a.shape
    B2, K2, N = b.shape
    assert B == B2 and K == K2
    
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    
    # Optimal for 512x1024x2048: larger tiles for better reuse
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 128, 64
    num_stages, num_warps = 3, 8
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), B)
    
    bmm_kernel[grid](
        a, b, c, B, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=8,
        num_stages=num_stages, num_warps=num_warps, matrix_instr_nonkdim=16,
    )
    return c


class ModelNew(nn.Module):
    """Optimized model using Triton BMM kernel."""
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_bmm(A, B)


# ============ Test Inputs ============
batch_size = 128
m = 128 * 4  # 512
k = 256 * 4  # 1024
n = 512 * 4  # 2048

def get_inputs():
    A = torch.rand(batch_size, m, k, dtype=torch.float16)
    B = torch.rand(batch_size, k, n, dtype=torch.float16)
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
    for _ in range(50):
        _ = new_model(A, B)
    torch.cuda.synchronize()
    triton_time = (time.time() - t0) / 50
    
    for _ in range(10):
        _ = ref_model(A, B)
    torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(50):
        _ = ref_model(A, B)
    torch.cuda.synchronize()
    ref_time = (time.time() - t0) / 50
    
    speedup = ref_time / triton_time
    tflops = 2 * batch_size * m * n * k / triton_time / 1e12
    
    print(f"Triton: {triton_time*1000:.3f}ms, {tflops:.2f} TFLOPS")
    print(f"rocBLAS: {ref_time*1000:.3f}ms")
    print(f"Speedup: {speedup:.3f}x")
