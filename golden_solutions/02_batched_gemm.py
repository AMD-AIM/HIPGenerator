#!/usr/bin/env python3
"""
Problem: 3_Batched_matrix_multiplication
Shape: (128, 512, 1024) x (128, 1024, 2048) -> (128, 512, 2048)
Target: MI350 (gfx950), 32 XCDs, 256 CUs

Optimizations:
- 2D grid (tiles, batch) for parallelism
- Pingpong scheduling
- 16x16 MFMA
- 256x256x32 tiles, 3 stages
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
BATCH = 128
M_SIZE = 512
K_SIZE = 1024
N_SIZE = 2048

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
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model with minimized launch overhead.
    Precomputes grid, strides, and preallocates output buffer.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Kernel config
        self._BLOCK_M, self._BLOCK_N, self._BLOCK_K = 256, 256, 32
        self._num_stages, self._num_warps = 3, 8
        self._GROUP_M = 8
        # Precompute grid (2D: tiles, batch)
        self._grid = (triton.cdiv(M_SIZE, self._BLOCK_M) * triton.cdiv(N_SIZE, self._BLOCK_N), BATCH)
        # Preallocate output buffer
        self.register_buffer('_C', torch.empty((BATCH, M_SIZE, N_SIZE), dtype=torch.float16))
        # Precompute strides for contiguous row-major 3D tensors
        # A[B, M, K]: stride = (M*K, K, 1)
        self._stride_ab = M_SIZE * K_SIZE
        self._stride_am = K_SIZE
        self._stride_ak = 1
        # B[B, K, N]: stride = (K*N, N, 1)
        self._stride_bb = K_SIZE * N_SIZE
        self._stride_bk = N_SIZE
        self._stride_bn = 1
        # C[B, M, N]: stride = (M*N, N, 1)
        self._stride_cb = M_SIZE * N_SIZE
        self._stride_cm = N_SIZE
        self._stride_cn = 1
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        bmm_kernel[self._grid](
            A, B, self._C, BATCH, M_SIZE, N_SIZE, K_SIZE,
            self._stride_ab, self._stride_am, self._stride_ak,
            self._stride_bb, self._stride_bk, self._stride_bn,
            self._stride_cb, self._stride_cm, self._stride_cn,
            BLOCK_M=self._BLOCK_M, BLOCK_N=self._BLOCK_N, BLOCK_K=self._BLOCK_K,
            GROUP_M=self._GROUP_M,
            num_stages=self._num_stages, num_warps=self._num_warps, matrix_instr_nonkdim=16,
        )
        return self._C


def get_inputs():
    A = torch.rand(BATCH, M_SIZE, K_SIZE, dtype=torch.float16)
    B = torch.rand(BATCH, K_SIZE, N_SIZE, dtype=torch.float16)
    return [A, B]

def get_init_inputs():
    return []


# ============ Verification ============
if __name__ == "__main__":
    import time
    
    A = torch.rand(BATCH, M_SIZE, K_SIZE, dtype=torch.float16, device='cuda')
    B = torch.rand(BATCH, K_SIZE, N_SIZE, dtype=torch.float16, device='cuda')
    
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
    tflops = 2 * BATCH * M_SIZE * N_SIZE * K_SIZE / triton_time / 1e12
    
    print(f"Triton: {triton_time*1000:.3f}ms, {tflops:.2f} TFLOPS")
    print(f"rocBLAS: {ref_time*1000:.3f}ms")
    print(f"Speedup: {speedup:.3f}x")
