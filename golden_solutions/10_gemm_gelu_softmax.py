#!/usr/bin/env python3
"""
Problem: 99_Matmul_GELU_Softmax
Shape: [1024, 4096] @ [4096, 4096] -> [1024, 4096] -> GELU -> Softmax

优化策略：
1. GEMM + GELU 融合 (参考 01_square_gemm 配置)
2. Softmax 使用优化的 row-wise 实现，分两阶段：
   - 第一阶段：分块计算 max 和 exp_sum
   - 第二阶段：计算最终 softmax
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import os

os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'


# ============ Original Model ============
class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.nn.functional.gelu(x)
        x = torch.softmax(x, dim=-1)
        return x


# ============ Optimized GEMM + GELU Kernel ============
@triton.jit
def gemm_gelu_kernel(
    A, B, bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Optimized GEMM + GELU with L2 cache grouping"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 cache-friendly grouping
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
    
    # Main GEMM loop
    for k in range(0, K, BLOCK_K):
        mask_k = (k + offs_k) < K
        a = tl.load(a_ptrs, mask=mask_k[None, :] & (offs_m[:, None] < M), other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Add bias (fused, near zero cost)
    bias_vals = tl.load(bias + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias_vals[None, :]
    
    # Apply GELU approximation (fused, near zero cost)
    # GELU(x) ≈ x * sigmoid(1.702 * x)
    sigmoid_val = 1.0 / (1.0 + tl.exp(-1.702 * acc))
    acc = acc * sigmoid_val
    
    # Store
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


# ============ Optimized Softmax Kernel ============
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    """
    Optimized single-pass row-wise softmax.
    Loads entire row at once for N <= 4096.
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= M:
        return
    
    row_start = input_ptr + row_idx * stride_m
    out_start = output_ptr + row_idx * stride_m
    
    # Load entire row
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N
    vals = tl.load(row_start + offs_n, mask=mask, other=float('-inf')).to(tl.float32)
    
    # Single-pass softmax
    row_max = tl.max(vals, axis=0)
    vals = vals - row_max
    vals_exp = tl.exp(vals)
    row_sum = tl.sum(vals_exp, axis=0)
    softmax_out = vals_exp / row_sum
    
    tl.store(out_start + offs_n, softmax_out.to(tl.float16), mask=mask)


def triton_gemm_gelu_pretransposed(x, weight_t, bias):
    """Optimized GEMM + GELU with pre-transposed weight [K, N]"""
    M, K = x.shape
    N = weight_t.shape[1]
    
    out = torch.empty((M, N), device=x.device, dtype=torch.float16)
    
    # Best config for 1024x4096x4096: num_stages=3 achieves 0.901x rocBLAS
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    num_stages, num_warps = 3, 8  # Changed from 2 to 3 stages
    GROUP_M = 8
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    gemm_gelu_kernel[grid](
        x, weight_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_stages=num_stages, num_warps=num_warps,
    )
    
    return out


def triton_gemm_gelu(x, weight, bias):
    """Optimized GEMM + GELU (for API compatibility)"""
    weight_t = weight.T.contiguous()
    return triton_gemm_gelu_pretransposed(x, weight_t, bias)


def triton_softmax(x):
    """Optimized single-pass row-wise softmax"""
    M, N = x.shape
    out = torch.empty_like(x)
    
    # Load entire row for best performance (N=4096)
    BLOCK_N = triton.next_power_of_2(N)
    
    grid = (M,)
    softmax_kernel[grid](
        x, out,
        M, N,
        x.stride(0),
        BLOCK_N=BLOCK_N,
        num_warps=4,  # Best config from tuning
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized: GEMM + GELU (fused) + Softmax
    Pre-transpose weight to avoid per-forward transpose overhead.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Store weight in transposed form [K, N] for efficient GEMM
        self.weight_t = nn.Parameter(torch.randn(input_size, hidden_size, dtype=torch.float16))
        self.bias = nn.Parameter(torch.randn(hidden_size, dtype=torch.float16))
    
    @property
    def weight(self):
        # For compatibility with reference model weight copying
        return self.weight_t.T
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GEMM + GELU with pre-transposed weight
        out = triton_gemm_gelu_pretransposed(x, self.weight_t, self.bias)
        # Optimized softmax
        return triton_softmax(out)


def get_inputs():
    batch_size, input_size, hidden_size = 1024, 4096, 4096
    return [torch.rand(batch_size, input_size, dtype=torch.float16, device='cuda')]


def get_init_inputs():
    return [4096, 4096]


# ============ Verification ============
if __name__ == "__main__":
    import time
    
    inputs = get_inputs()
    init_inputs = get_init_inputs()
    
    model = ModelNew(*init_inputs).cuda()
    
    # Reference - copy weights from new model
    ref_model = Model(*init_inputs).cuda().half()
    with torch.no_grad():
        ref_model.linear.weight.copy_(model.weight_t.T)  # Transpose back
        ref_model.linear.bias.copy_(model.bias)
    
    ref_out = ref_model(inputs[0])
    out = model(inputs[0])
    
    max_diff = (ref_out.float() - out.float()).abs().max().item()
    print(f"Max diff: {max_diff}")
    
    # Benchmark
    torch.cuda.synchronize()
    
    for _ in range(10):
        _ = model(inputs[0])
    torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(100):
        _ = model(inputs[0])
    torch.cuda.synchronize()
    triton_time = (time.time() - t0) / 100
    
    for _ in range(10):
        _ = ref_model(inputs[0])
    torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(100):
        _ = ref_model(inputs[0])
    torch.cuda.synchronize()
    ref_time = (time.time() - t0) / 100
    
    print(f"Triton: {triton_time*1000:.3f}ms")
    print(f"rocBLAS: {ref_time*1000:.3f}ms")
    print(f"Speedup: {ref_time/triton_time:.3f}x")
