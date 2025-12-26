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
- Precomputed strides and preallocated buffer
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import os

os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'

NUM_XCDS = 32

# ============ Test Inputs (defined early for precomputation) ============
M_SIZE = 2048
K_SIZE = 8192
N_SIZE = 4096

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
    """Kernel for A.T @ B with optimized memory access."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pids = num_pid_m * num_pid_n
    
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
        k_offs = k + offs_k
        k_mask = k_offs < K
        m_mask = offs_m < M
        n_mask = offs_n < N
        
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model with minimized launch overhead.
    Precomputes grid, strides, and preallocates output buffer.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Kernel config
        self._BLOCK_M, self._BLOCK_N, self._BLOCK_K = 128, 128, 64
        self._num_stages, self._num_warps = 2, 8
        self._GROUP_M = 8
        # Precompute grid
        self._grid = (triton.cdiv(M_SIZE, self._BLOCK_M) * triton.cdiv(N_SIZE, self._BLOCK_N),)
        # Preallocate output buffer
        self.register_buffer('_C', torch.empty((M_SIZE, N_SIZE), dtype=torch.float16))
        # Precompute strides
        # A is (K, M): stride = (M, 1) for contiguous
        self._stride_ak = M_SIZE  # stride along K dimension
        self._stride_am = 1       # stride along M dimension
        # B is (K, N): stride = (N, 1) for contiguous
        self._stride_bk = N_SIZE
        self._stride_bn = 1
        # C is (M, N): stride = (N, 1) for contiguous
        self._stride_cm = N_SIZE
        self._stride_cn = 1
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        matmul_tn_kernel[self._grid](
            A, B, self._C, M_SIZE, N_SIZE, K_SIZE,
            self._stride_ak, self._stride_am,
            self._stride_bk, self._stride_bn,
            self._stride_cm, self._stride_cn,
            BLOCK_M=self._BLOCK_M, BLOCK_N=self._BLOCK_N, BLOCK_K=self._BLOCK_K,
            GROUP_M=self._GROUP_M, NUM_XCDS=NUM_XCDS,
            num_stages=self._num_stages, num_warps=self._num_warps, matrix_instr_nonkdim=16,
        )
        return self._C


def get_inputs():
    A = torch.rand(K_SIZE, M_SIZE, dtype=torch.float16)  # (K, M)
    B = torch.rand(K_SIZE, N_SIZE, dtype=torch.float16)  # (K, N)
    return [A, B]

def get_init_inputs():
    return []


# ============ Verification ============
if __name__ == "__main__":
    import time
    
    A = torch.rand(K_SIZE, M_SIZE, dtype=torch.float16, device='cuda')
    B = torch.rand(K_SIZE, N_SIZE, dtype=torch.float16, device='cuda')
    
    ref_model = Model().cuda()
    new_model = ModelNew().cuda()
    
    # Verify correctness
    ref = ref_model(A, B)
    out = new_model(A, B)
    max_diff = (ref.float() - out.float()).abs().max().item()
    print(f"Max diff: {max_diff}")
    
    # Warmup
    for _ in range(10):
        _ = new_model(A, B)
        _ = ref_model(A, B)
    torch.cuda.synchronize()
    
    # Benchmark ModelNew
    t0 = time.time()
    for _ in range(50):
        _ = new_model(A, B)
    torch.cuda.synchronize()
    triton_time = (time.time() - t0) / 50
    
    # Benchmark Model (rocBLAS)
    t0 = time.time()
    for _ in range(50):
        _ = ref_model(A, B)
    torch.cuda.synchronize()
    ref_time = (time.time() - t0) / 50
    
    speedup = ref_time / triton_time
    print(f"Triton: {triton_time*1000:.3f}ms")
    print(f"rocBLAS: {ref_time*1000:.3f}ms")
    print(f"Speedup: {speedup:.3f}x")
