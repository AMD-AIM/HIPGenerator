import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 4096
dim = 393216

# Original PyTorch Model (reference for evaluation)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sigmoid(x) * x  # Swish activation

def get_inputs():
    x = torch.rand(batch_size, dim, dtype=torch.float16, device='cuda')
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed

# ============================================================
# Generated TRITON Kernel (12/29/2025, 9:07:16 AM)
# ============================================================

@triton.jit
def swish_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    # Swish: x * sigmoid(x)
    x_f32 = x.to(tl.float32)
    sigmoid_x = tl.sigmoid(x_f32)
    result = x_f32 * sigmoid_x
    tl.store(output_ptr + offsets, result.to(tl.float16), mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve shape for reshape at the end
        original_shape = x.shape
        x_flat = x.contiguous().view(-1)
        
        # Allocate output with same dtype as input
        output = torch.empty_like(x_flat)
        
        n_elements = x_flat.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        swish_kernel[grid](
            x_flat, output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output.view(original_shape)
