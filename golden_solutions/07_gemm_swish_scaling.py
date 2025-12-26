#!/usr/bin/env python3
"""
Problem: 59_Matmul_Swish_Scaling
Shape: x(batch, features) @ W.T + bias -> Swish -> Scale
Fusion: GEMM + BiasAdd + Swish + Scaling in single kernel
"""

import torch
import triton
import triton.language as tl
import os

os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'


@triton.jit
def gemm_swish_scaling_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K, scale,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Fused GEMM + BiasAdd + Swish + Scaling kernel."""
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
    
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        m_mask = offs_m < M
        n_mask = offs_n < N
        
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(x, w, acc)
        
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    # Swish: x * sigmoid(x) = x / (1 + exp(-x))
    acc = acc * tl.sigmoid(acc)
    
    # Scaling
    acc = acc * scale
    
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def triton_gemm_swish_scaling(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, scale: float) -> torch.Tensor:
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    num_stages, num_warps = 2, 8
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    gemm_swish_scaling_kernel[grid](
        x, weight, bias, out,
        M, N, K, scale,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=8,
        num_stages=num_stages, num_warps=num_warps, matrix_instr_nonkdim=16,
    )
    return out


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
        self.bias = torch.nn.Parameter(torch.randn(out_features, dtype=torch.float16))
        self.scaling_factor = scaling_factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_gemm_swish_scaling(x, self.weight, self.bias, self.scaling_factor)


# ============ Verification ============
if __name__ == "__main__":
    batch_size, in_features, out_features, scaling = 1024, 4096, 4096, 0.5
    
    x = torch.rand(batch_size, in_features, dtype=torch.float16, device='cuda')
    
    model = ModelNew(in_features, out_features, scaling).cuda()
    
    # Reference
    linear_out = torch.nn.functional.linear(x, model.weight, model.bias)
    swish_out = linear_out * torch.sigmoid(linear_out)
    ref = swish_out * scaling
    out = model(x)
    
    max_diff = (ref.float() - out.float()).abs().max().item()
    print(f"Max diff: {max_diff}")
    
    # Benchmark
    import time
    torch.cuda.synchronize()
    
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(50):
        _ = model(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - t0) / 50
    
    for _ in range(10):
        linear_out = torch.nn.functional.linear(x, model.weight, model.bias)
        _ = linear_out * torch.sigmoid(linear_out) * scaling
    torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(50):
        linear_out = torch.nn.functional.linear(x, model.weight, model.bias)
        _ = linear_out * torch.sigmoid(linear_out) * scaling
    torch.cuda.synchronize()
    ref_time = (time.time() - t0) / 50
    
    speedup = ref_time / triton_time
    print(f"Triton (fused): {triton_time*1000:.3f}ms")
    print(f"PyTorch (separate): {ref_time*1000:.3f}ms")
    print(f"Speedup: {speedup:.3f}x")


