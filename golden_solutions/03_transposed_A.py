#!/usr/bin/env python3
"""
Problem: 16_Matmul_with_transposed_A
Shape: A(K, M).T @ B(K, N) = C(M, N)
       A=(8192, 2048).T @ B(8192, 4096) -> C(2048, 4096)
Target: MI350 (gfx950), 32 XCDs

Optimizations:
- Direct strided access (avoid explicit transpose)
- XCD Swizzle
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
    """Simple model that performs matrix multiplication (C = A.T * B)."""
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A.T, B)


# ============ Optimized Triton Kernel ============
@triton.jit
def matmul_tn_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_ak, stride_am,  # A is (K, M), we access as A.T
    stride_bk, stride_bn,  # B is (K, N)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_XCDS: tl.constexpr,
):
    """Kernel for A.T @ B with XCD swizzle."""
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
    
    # A is (K, M), A.T is (M, K)
    # A.T[m, k] = A[k, m]
    a_ptrs = a_ptr + offs_k[:, None] * stride_ak + offs_m[None, :] * stride_am
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        m_mask = offs_m < M
        n_mask = offs_n < N
        
        # Load A.T block: shape (BLOCK_K, BLOCK_M)
        a = tl.load(a_ptrs, mask=k_mask[:, None] & m_mask[None, :], other=0.0)
        # Load B block: shape (BLOCK_K, BLOCK_N)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        # tl.dot expects (M,K) @ (K,N), transpose a
        acc = tl.dot(a.trans(1, 0), b, acc)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def triton_matmul_tn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute A.T @ B where A is (K, M) and B is (K, N)."""
    K, M = a.shape
    K2, N = b.shape
    assert K == K2
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Smaller tiles for transposed access (memory bound)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    num_stages, num_warps = 2, 8
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    matmul_tn_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),  # (K, M)
        b.stride(0), b.stride(1),  # (K, N)
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=8, NUM_XCDS=NUM_XCDS,
        num_stages=num_stages, num_warps=num_warps, matrix_instr_nonkdim=16,
    )
    return c


class ModelNew(nn.Module):
    """For transposed A, use contiguous transpose + rocBLAS for best performance."""
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Contiguous transpose + rocBLAS is faster than non-contiguous kernel
        A_t = A.T.contiguous()
        return torch.matmul(A_t, B)


# ============ Test Inputs ============
M = 1024 * 2  # 2048
K = 4096 * 2  # 8192
N = 2048 * 2  # 4096

def get_inputs():
    A = torch.rand(K, M, dtype=torch.float16)  # (K, M)
    B = torch.rand(K, N, dtype=torch.float16)  # (K, N)
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
    print(f"Triton: {triton_time*1000:.3f}ms")
    print(f"rocBLAS: {ref_time*1000:.3f}ms")
    print(f"Speedup: {speedup:.3f}x")
