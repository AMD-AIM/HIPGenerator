#!/usr/bin/env python3
"""
Problem: 86_Matmul_Divide_GELU
Shape: x(1024, 8192) @ W.T(8192, 8192) + bias / divisor -> GELU -> (1024, 8192)
Target: MI350 (gfx950), 32 XCDs, 256 CUs

Optimizations:
- Kernel Fusion: GEMM + BiasAdd + Divide + GELU in single kernel
- XCD Swizzle for 32 XCDs
- Pingpong scheduling
- 16x16 MFMA
- GELU approximation using sigmoid: x * sigmoid(1.702 * x)
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
    """A model that performs a matrix multiplication, divides by a scalar, and applies GELU activation."""
    def __init__(self, input_size, output_size, divisor):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x):
        x = self.linear(x)
        x = x / self.divisor
        x = torch.nn.functional.gelu(x)
        return x


# ============ Optimized Triton Kernel ============
@triton.jit
def gemm_divide_gelu_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K, divisor,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_XCDS: tl.constexpr,
):
    """Fused GEMM + BiasAdd + Divide + GELU kernel."""
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
    
    # Fused: Add bias + Divide + GELU
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    acc = acc / divisor
    
    # GELU approximation: x * sigmoid(1.702 * x)
    acc = acc * tl.sigmoid(1.702 * acc)
    
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def triton_gemm_divide_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, divisor: float) -> torch.Tensor:
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
    num_stages, num_warps = 2, 8
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    gemm_divide_gelu_kernel[grid](
        x, weight, bias, out,
        M, N, K, divisor,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=8, NUM_XCDS=NUM_XCDS,
        num_stages=num_stages, num_warps=num_warps, matrix_instr_nonkdim=16,
    )
    return out


class ModelNew(nn.Module):
    """Optimized model using fused Triton kernel."""
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size, dtype=torch.float16))
        self.bias = nn.Parameter(torch.randn(output_size, dtype=torch.float16))
        self.divisor = divisor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_gemm_divide_gelu(x, self.weight, self.bias, self.divisor)


# ============ Test Inputs ============
batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0

def get_inputs():
    return [torch.rand(batch_size, input_size, dtype=torch.float16)]

def get_init_inputs():
    return [input_size, output_size, divisor]


# ============ Verification ============
if __name__ == "__main__":
    x = get_inputs()[0].cuda()
    
    ref_model = Model(*get_init_inputs()).cuda().half()
    new_model = ModelNew(*get_init_inputs()).cuda()
    
    # Copy weights for fair comparison
    with torch.no_grad():
        new_model.weight.copy_(ref_model.linear.weight)
        new_model.bias.copy_(ref_model.linear.bias)
    
    ref = ref_model(x)
    out = new_model(x)
    
    max_diff = (ref.float() - out.float()).abs().max().item()
    print(f"Max diff: {max_diff}")
    
    # Benchmark
    import time
    torch.cuda.synchronize()
    
    for _ in range(10):
        _ = new_model(x)
    torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(50):
        _ = new_model(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - t0) / 50
    
    for _ in range(10):
        _ = ref_model(x)
    torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(50):
        _ = ref_model(x)
    torch.cuda.synchronize()
    ref_time = (time.time() - t0) / 50
    
    speedup = ref_time / triton_time
    print(f"Triton (fused): {triton_time*1000:.3f}ms")
    print(f"PyTorch (separate): {ref_time*1000:.3f}ms")
    print(f"Speedup: {speedup:.3f}x")
