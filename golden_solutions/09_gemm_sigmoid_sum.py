#!/usr/bin/env python3
"""
Problem: 56_Matmul_Sigmoid_Sum
Shape: [1024, 4096] @ [4096, 4096] -> [1024, 4096] -> sigmoid -> sum(dim=1)
Target: MI350 (gfx950), 32 XCDs

Optimizations:
1. Fused GEMM + Sigmoid kernel
2. Optimized row-wise sum reduction
3. Precomputed strides and preallocated buffers
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import os

os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'

# ============ Test Inputs (defined early for precomputation) ============
M_SIZE = 1024
K_SIZE = 4096
N_SIZE = 4096


# ============ Original Model ============
class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.sigmoid(x)
        x = x.sum(dim=1)
        return x


# ============ Optimized GEMM + Sigmoid Kernel ============
@triton.jit
def gemm_sigmoid_kernel(
    A, B, bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Optimized GEMM + Sigmoid with L2 cache grouping"""
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
    
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        mask_k = (k + offs_k) < K
        a = tl.load(a_ptrs, mask=mask_k[None, :] & (offs_m[:, None] < M), other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Fused: Add bias + Sigmoid
    bias_vals = tl.load(bias + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias_vals[None, :]
    acc = 1.0 / (1.0 + tl.exp(-acc))
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


# ============ Optimized Row-wise Sum Kernel ============
@triton.jit
def row_sum_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_m,
    BLOCK_N: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    """Optimized row-wise sum with coalesced memory access."""
    row_idx = tl.program_id(0)
    
    if row_idx >= M:
        return
    
    row_start = input_ptr + row_idx * stride_m
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    for block_start in range(0, N, BLOCK_N):
        offs_n = block_start + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        vals = tl.load(row_start + offs_n, mask=mask, other=0.0)
        acc += vals.to(tl.float32)
    
    row_sum = tl.sum(acc, axis=0)
    out_offs = tl.arange(0, 1)
    tl.store(output_ptr + row_idx + out_offs, row_sum, mask=(out_offs == 0))


class ModelNew(nn.Module):
    """
    Optimized model with minimized launch overhead.
    Pre-transpose weight, precompute grid/strides, preallocate buffers.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Store weight in transposed form [K, N] for efficient GEMM
        self.weight_t = nn.Parameter(torch.randn(input_size, hidden_size, dtype=torch.float16))
        self.bias = nn.Parameter(torch.randn(hidden_size, dtype=torch.float16))
        
        # GEMM kernel config
        self._BLOCK_M, self._BLOCK_N, self._BLOCK_K = 128, 128, 64
        self._num_stages, self._num_warps = 3, 8
        self._GROUP_M = 8
        self._gemm_grid = (triton.cdiv(M_SIZE, self._BLOCK_M) * triton.cdiv(N_SIZE, self._BLOCK_N),)
        
        # Sum kernel config
        self._sum_BLOCK_N = 256
        self._sum_NUM_BLOCKS = triton.cdiv(N_SIZE, self._sum_BLOCK_N)
        self._sum_grid = (M_SIZE,)
        
        # Preallocate buffers
        self.register_buffer('_gemm_out', torch.empty((M_SIZE, N_SIZE), dtype=torch.float16))
        self.register_buffer('_sum_out', torch.empty(M_SIZE, dtype=torch.float32))
        
        # Precompute strides
        self._stride_am = K_SIZE
        self._stride_ak = 1
        self._stride_bk = N_SIZE
        self._stride_bn = 1
        self._stride_cm = N_SIZE
        self._stride_cn = 1
    
    @property
    def weight(self):
        return self.weight_t.T
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GEMM + Sigmoid
        gemm_sigmoid_kernel[self._gemm_grid](
            x, self.weight_t, self.bias, self._gemm_out,
            M_SIZE, N_SIZE, K_SIZE,
            self._stride_am, self._stride_ak,
            self._stride_bk, self._stride_bn,
            self._stride_cm, self._stride_cn,
            BLOCK_M=self._BLOCK_M, BLOCK_N=self._BLOCK_N, BLOCK_K=self._BLOCK_K,
            GROUP_M=self._GROUP_M,
            num_stages=self._num_stages, num_warps=self._num_warps,
        )
        
        # Row Sum
        row_sum_kernel[self._sum_grid](
            self._gemm_out, self._sum_out,
            M_SIZE, N_SIZE,
            self._stride_cm,
            BLOCK_N=self._sum_BLOCK_N,
            NUM_BLOCKS=self._sum_NUM_BLOCKS,
            num_warps=4,
        )
        
        return self._sum_out


def get_inputs():
    return [torch.rand(M_SIZE, K_SIZE, dtype=torch.float16, device='cuda')]

def get_init_inputs():
    return [K_SIZE, N_SIZE]


# ============ Verification ============
if __name__ == "__main__":
    import time
    
    inputs = get_inputs()
    init_inputs = get_init_inputs()
    
    ref_model = Model(*init_inputs).cuda().half()
    new_model = ModelNew(*init_inputs).cuda()
    
    # Copy weights for fair comparison
    with torch.no_grad():
        new_model.weight_t.copy_(ref_model.linear.weight.T)
        new_model.bias.copy_(ref_model.linear.bias)
    
    # Verify correctness
    ref = ref_model(inputs[0]).float()
    out = new_model(inputs[0]).float()
    max_diff = (ref - out).abs().max().item()
    print(f"Max diff: {max_diff}")
    
    # Warmup
    for _ in range(10):
        _ = new_model(inputs[0])
        _ = ref_model(inputs[0])
    torch.cuda.synchronize()
    
    # Benchmark ModelNew
    t0 = time.time()
    for _ in range(100):
        _ = new_model(inputs[0])
    torch.cuda.synchronize()
    triton_time = (time.time() - t0) / 100
    
    # Benchmark Model (rocBLAS)
    t0 = time.time()
    for _ in range(100):
        _ = ref_model(inputs[0])
    torch.cuda.synchronize()
    ref_time = (time.time() - t0) / 100
    
    print(f"Triton: {triton_time*1000:.3f}ms")
    print(f"rocBLAS: {ref_time*1000:.3f}ms")
    print(f"Speedup: {ref_time/triton_time:.3f}x")
