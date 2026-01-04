import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

class Model(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

batch_size = 8192
dim = 8192

def get_inputs():
    return [torch.rand(batch_size, dim, dtype=torch.float16)]

def get_init_inputs():
    return []

# ============================================================
# Generated TRITON Kernel (12/31/2025, 3:27:03 AM)
# Evaluation: Accuracy=✓ Speedup=6.78x
# ============================================================

import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load and convert to float32 for numerical stability
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # GELU: 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))
    # sqrt(2.0 / pi) ≈ 0.7978845608028654
    x_cubed = x * x * x
    tanh_arg = 0.7978845608028654 * (x + 0.044715 * x_cubed)
    
    # Use sigmoid-based tanh implementation: tanh(x) = 2 * sigmoid(2*x) - 1
    tanh_val = 2.0 * tl.sigmoid(2.0 * tanh_arg) - 1.0
    
    y = 0.5 * x * (1.0 + tanh_val)
    
    # Convert back to input dtype and store
    tl.store(output_ptr + offsets, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x):
        # Preserve shape for reshape at the end
        original_shape = x.shape
        x_flat = x.contiguous().view(-1)
        
        # Allocate output with same dtype as input
        output = torch.empty_like(x_flat)
        
        n_elements = x_flat.numel()
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        gelu_kernel[grid](
            x_flat, output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output.view(original_shape)

# ============================================================
# Optimized Triton Kernel: ModelNewOptimized (12/31/2025, 3:28:07 AM)
# ModelNew → ModelNewOptimized
# Evaluation: Compile=✓ Accuracy=✓ Speedup=1.11x
# ============================================================

import os
os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def gelu_kernel_optimized(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load entire block at once and convert to float32 for numerical stability
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # GELU: 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))
    # Precomputed constant: sqrt(2.0 / pi) ≈ 0.7978845608028654
    x_squared = x * x
    x_cubed = x_squared * x
    
    # Fused computation
    tanh_arg = 0.7978845608028654 * (x + 0.044715 * x_cubed)
    
    # Use sigmoid-based tanh implementation: tanh(x) = 2 * sigmoid(2*x) - 1
    # This is more efficient on AMD GPUs
    tanh_val = 2.0 * tl.sigmoid(2.0 * tanh_arg) - 1.0
    
    # Final GELU computation
    y = 0.5 * x * (1.0 + tanh_val)
    
    # Convert back to float16 and store entire block at once
    tl.store(output_ptr + offsets, y.to(tl.float16), mask=mask)


class ModelNewOptimized(nn.Module):
    def __init__(self):
        super(ModelNewOptimized, self).__init__()
        # Precompute constants to reduce launch overhead
        self._sqrt_2_over_pi = 0.7978845608028654
        self._coeff = 0.044715
    
    def forward(self, x):
        # Preserve shape for reshape at the end
        original_shape = x.shape
        x_flat = x.contiguous().view(-1)
        
        # Preallocate output buffer
        output = torch.empty_like(x_flat)
        
        n_elements = x_flat.numel()
        
        # Use autotune to find optimal block size
        # Grid is computed by autotune based on BLOCK_SIZE
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        gelu_kernel_optimized[grid](
            x_flat, output,
            n_elements,
        )
        
        return output.view(original_shape)


batch_size = 8192
dim = 8192

def get_inputs():
    return [torch.rand(batch_size, dim, dtype=torch.float16)]

def get_init_inputs():
    return []