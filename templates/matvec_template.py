"""Pre-verified HIP Matrix-Vector multiplication template (memory-bound, no MFMA)"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_src = '''
#include <torch/extension.h>
torch::Tensor matvec_forward(torch::Tensor A, torch::Tensor x);
'''

hip_src = r'''
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

constexpr int BLOCK_SIZE = 256;
constexpr int ELEMENTS_PER_THREAD = 8;  // Process 8 bf16 values per thread

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__global__ void matvec_kernel(
    const hip_bfloat16* __restrict__ A,
    const hip_bfloat16* __restrict__ x,
    hip_bfloat16* __restrict__ y,
    int M, int K
) {
    __shared__ float smem[BLOCK_SIZE / 64];  // One value per warp
    
    int row = blockIdx.x;
    if (row >= M) return;
    
    const hip_bfloat16* A_row = A + row * K;
    
    float sum = 0.0f;
    
    // Each thread processes multiple elements with vectorized loads
    int tid = threadIdx.x;
    int stride = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    
    for (int k = tid * ELEMENTS_PER_THREAD; k < K; k += stride) {
        int remaining = min(ELEMENTS_PER_THREAD, K - k);
        
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            if (i < remaining) {
                float a_val = static_cast<float>(A_row[k + i]);
                float x_val = static_cast<float>(x[k + i]);
                sum += a_val * x_val;
            }
        }
    }
    
    // Warp reduction
    sum = warp_reduce_sum(sum);
    
    // Block reduction
    int warp_id = threadIdx.x / 64;
    int lane_id = threadIdx.x % 64;
    
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        int num_warps = BLOCK_SIZE / 64;
        float val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        
        if (lane_id == 0) {
            y[row] = hip_bfloat16(val);
        }
    }
}

torch::Tensor matvec_forward(torch::Tensor A, torch::Tensor x) {
    auto A_cont = A.contiguous();
    auto x_cont = x.contiguous().view(-1);  // Flatten x to 1D
    
    int M = A_cont.size(0);
    int K = A_cont.size(1);
    
    auto y = torch::empty({M, 1}, A_cont.options());
    
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);
    
    matvec_kernel<<<grid, block>>>(
        reinterpret_cast<const hip_bfloat16*>(A_cont.data_ptr<at::BFloat16>()),
        reinterpret_cast<const hip_bfloat16*>(x_cont.data_ptr<at::BFloat16>()),
        reinterpret_cast<hip_bfloat16*>(y.data_ptr<at::BFloat16>()),
        M, K
    );
    
    return y;
}
'''

module = load_inline(
    name="matvec_hip",
    cpp_sources=cpp_src,
    cuda_sources=hip_src,
    functions=["matvec_forward"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "-std=c++20",
                       "-I/opt/rocm/include/hip",
                       "--offload-arch=gfx950"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, A, x):
        return module.matvec_forward(A, x)

