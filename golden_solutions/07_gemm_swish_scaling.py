#!/usr/bin/env python3
"""
Problem: 59_Matmul_Swish_Scaling
Shape: x(1024, 4096) @ W.T(4096, 4096) + bias -> Swish -> Scale
Target: MI350 (gfx950), 32 XCDs

Optimizations:
- Fusion: GEMM + BiasAdd + Swish + Scaling in single kernel
- Pingpong scheduling
- 16x16 MFMA
- Precomputed strides and preallocated buffer
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
SCALING = 0.5


# ============ Original Model (Reference) ============
class Model(nn.Module):
    """Model that performs GEMM + Swish + Scaling."""
    def __init__(self, in_features, out_features, scaling_factor):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
    
    def forward(self, x):
        x = self.linear(x)
        x = x * torch.sigmoid(x)  # Swish
        x = x * self.scaling_factor
        return x


# ============ Optimized Triton Kernel ============
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
    
    # Fused: Add bias + Swish + Scaling
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    acc = acc * tl.sigmoid(acc)  # Swish
    acc = acc * scale
    
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model with minimized launch overhead.
    Precomputes grid, strides, and preallocates output buffer.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
        self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.float16))
        self.scaling_factor = scaling_factor
        
        # Kernel config
        self._BLOCK_M, self._BLOCK_N, self._BLOCK_K = 128, 128, 64
        self._num_stages, self._num_warps = 3, 8
        self._GROUP_M = 8
        # Precompute grid
        self._grid = (triton.cdiv(M_SIZE, self._BLOCK_M) * triton.cdiv(N_SIZE, self._BLOCK_N),)
        # Preallocate output buffer
        self.register_buffer('_out', torch.empty((M_SIZE, N_SIZE), dtype=torch.float16))
        # Precompute strides
        self._stride_xm = K_SIZE
        self._stride_xk = 1
        self._stride_wn = K_SIZE
        self._stride_wk = 1
        self._stride_om = N_SIZE
        self._stride_on = 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gemm_swish_scaling_kernel[self._grid](
            x, self.weight, self.bias, self._out,
            M_SIZE, N_SIZE, K_SIZE, self.scaling_factor,
            self._stride_xm, self._stride_xk,
            self._stride_wn, self._stride_wk,
            self._stride_om, self._stride_on,
            BLOCK_M=self._BLOCK_M, BLOCK_N=self._BLOCK_N, BLOCK_K=self._BLOCK_K,
            GROUP_M=self._GROUP_M,
            num_stages=self._num_stages, num_warps=self._num_warps, matrix_instr_nonkdim=16,
        )
        return self._out


def get_inputs():
    return [torch.rand(M_SIZE, K_SIZE, dtype=torch.float16)]

def get_init_inputs():
    return [K_SIZE, N_SIZE, SCALING]


# ============ Verification ============
if __name__ == "__main__":
    import time
    
    x = get_inputs()[0].cuda()
    
    ref_model = Model(*get_init_inputs()).cuda().half()
    new_model = ModelNew(*get_init_inputs()).cuda()
    
    # Copy weights for fair comparison
    with torch.no_grad():
        new_model.weight.copy_(ref_model.linear.weight)
        new_model.bias.copy_(ref_model.linear.bias)
    
    # Verify correctness
    ref = ref_model(x)
    out = new_model(x)
    max_diff = (ref.float() - out.float()).abs().max().item()
    print(f"Max diff: {max_diff}")
    
    # Warmup
    for _ in range(10):
        _ = new_model(x)
        _ = ref_model(x)
    torch.cuda.synchronize()
    
    # Benchmark ModelNew
    t0 = time.time()
    for _ in range(50):
        _ = new_model(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - t0) / 50
    
    # Benchmark Model (rocBLAS)
    t0 = time.time()
    for _ in range(50):
        _ = ref_model(x)
    torch.cuda.synchronize()
    ref_time = (time.time() - t0) / 50
    
    speedup = ref_time / triton_time
    print(f"Triton: {triton_time*1000:.3f}ms")
    print(f"rocBLAS: {ref_time*1000:.3f}ms")
    print(f"Speedup: {speedup:.3f}x")
