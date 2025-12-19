"""Pre-verified HipKittens GEMM template - DO NOT MODIFY"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_src = '''
#include <torch/extension.h>
torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B);
'''

hip_src = r'''
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>
#include "kittens.cuh"

using namespace kittens;

constexpr int BLOCK_SIZE = 256;
constexpr int HALF_BLOCK_SIZE = 128;  
constexpr int K_STEP = 64;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;
constexpr int NUM_THREADS = kittens::WARP_THREADS * NUM_WARPS;
constexpr int REG_BLOCK_M = BLOCK_SIZE / WARPS_M;
constexpr int REG_BLOCK_N = BLOCK_SIZE / WARPS_N;
constexpr int HALF_REG_BLOCK_M = REG_BLOCK_M / 2;
constexpr int HALF_REG_BLOCK_N = REG_BLOCK_N / 2;

using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;
using G = kittens::group<NUM_WARPS>;

__global__ __launch_bounds__(NUM_THREADS, 2)
void gemm_mfma_kernel(_gl_bf16 g_A, _gl_bf16 g_B, _gl_bf16 g_C, int M, int K, int N) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    using ST_A = st_bf<HALF_BLOCK_SIZE, K_STEP, st_16x32_s>;
    using ST_B = st_bf<HALF_BLOCK_SIZE, K_STEP, st_16x32_s>;
    ST_A (&As)[2][2] = al.allocate<ST_A, 2, 2>();
    ST_B (&Bs)[2][2] = al.allocate<ST_B, 2, 2>();
    
    rt_bf<HALF_REG_BLOCK_M, K_STEP, row_l, rt_16x32_s> A_tile;
    rt_bf<HALF_REG_BLOCK_N, K_STEP, row_l, rt_16x32_s> B_tile_0, B_tile_1;
    rt_fl<HALF_REG_BLOCK_M, HALF_REG_BLOCK_N, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);
    
    int wgid = blockIdx.x;
    const int num_pid_m = M / BLOCK_SIZE;
    const int num_pid_n = N / BLOCK_SIZE;
    constexpr int WGM = 8;
    const int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    int row = pid_m, col = pid_n;
    
    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    const int num_tiles = K / K_STEP;
    
    using T = typename ST_A::dtype;
    constexpr int bytes_per_thread = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * K_STEP * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile/2];
    uint32_t swizzled_offsets_B[memcpy_per_tile/2];
    G::prefill_swizzled_offsets(As[0][0], g_A, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0], g_B, swizzled_offsets_B);
    
    G::load(As[0][0], g_A, {0, 0, row*2, 0}, swizzled_offsets_A);
    G::load(As[0][1], g_A, {0, 0, row*2+1, 0}, swizzled_offsets_A);
    G::load(Bs[0][0], g_B, {0, 0, col*2, 0}, swizzled_offsets_B);
    G::load(Bs[0][1], g_B, {0, 0, col*2+1, 0}, swizzled_offsets_B);
    __builtin_amdgcn_s_barrier();
    
    int tic = 0, toc = 1;
    
    for (int tile = 0; tile < num_tiles - 1; tile++) {
        G::load(As[toc][0], g_A, {0, 0, row*2, tile+1}, swizzled_offsets_A);
        G::load(As[toc][1], g_A, {0, 0, row*2+1, tile+1}, swizzled_offsets_A);
        G::load(Bs[toc][0], g_B, {0, 0, col*2, tile+1}, swizzled_offsets_B);
        G::load(Bs[toc][1], g_B, {0, 0, col*2+1, tile+1}, swizzled_offsets_B);
        
        auto st_a0 = subtile_inplace<HALF_REG_BLOCK_M, K_STEP>(As[tic][0], {warp_row, 0});
        auto st_a1 = subtile_inplace<HALF_REG_BLOCK_M, K_STEP>(As[tic][1], {warp_row, 0});
        auto st_b0 = subtile_inplace<HALF_REG_BLOCK_N, K_STEP>(Bs[tic][0], {warp_col, 0});
        auto st_b1 = subtile_inplace<HALF_REG_BLOCK_N, K_STEP>(Bs[tic][1], {warp_col, 0});
        
        load(A_tile, st_a0);
        load(B_tile_0, st_b0);
        load(B_tile_1, st_b1);
        __builtin_amdgcn_s_barrier();
        
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        
        load(A_tile, st_a1);
        __builtin_amdgcn_s_barrier();
        
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        
        tic ^= 1; toc ^= 1;
    }
    
    {
        auto st_a0 = subtile_inplace<HALF_REG_BLOCK_M, K_STEP>(As[tic][0], {warp_row, 0});
        auto st_a1 = subtile_inplace<HALF_REG_BLOCK_M, K_STEP>(As[tic][1], {warp_row, 0});
        auto st_b0 = subtile_inplace<HALF_REG_BLOCK_N, K_STEP>(Bs[tic][0], {warp_col, 0});
        auto st_b1 = subtile_inplace<HALF_REG_BLOCK_N, K_STEP>(Bs[tic][1], {warp_col, 0});
        
        load(A_tile, st_a0);
        load(B_tile_0, st_b0);
        load(B_tile_1, st_b1);
        __builtin_amdgcn_s_barrier();
        
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        
        load(A_tile, st_a1);
        __builtin_amdgcn_s_barrier();
        
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
    }
    
    store(g_C, C_accum[0][0], {0, 0, row*2*WARPS_M + warp_row, col*2*WARPS_N + warp_col});
    store(g_C, C_accum[0][1], {0, 0, row*2*WARPS_M + warp_row, col*2*WARPS_N + WARPS_N + warp_col});
    store(g_C, C_accum[1][0], {0, 0, row*2*WARPS_M + WARPS_M + warp_row, col*2*WARPS_N + warp_col});
    store(g_C, C_accum[1][1], {0, 0, row*2*WARPS_M + WARPS_M + warp_row, col*2*WARPS_N + WARPS_N + warp_col});
}

torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
    auto A_cont = A.contiguous();
    auto B_cont = B.contiguous();
    
    int M = A_cont.size(0);
    int K = A_cont.size(1);
    int N = B_cont.size(1);
    
    auto Bt = B_cont.t().contiguous();
    auto C = torch::empty({M, N}, A_cont.options());
    
    _gl_bf16 gA{(bf16*)A_cont.data_ptr<at::BFloat16>(), 1u, 1u, (unsigned)M, (unsigned)K};
    _gl_bf16 gB{(bf16*)Bt.data_ptr<at::BFloat16>(), 1u, 1u, (unsigned)N, (unsigned)K};
    _gl_bf16 gC{(bf16*)C.data_ptr<at::BFloat16>(), 1u, 1u, (unsigned)M, (unsigned)N};
    
    int num_blocks = (M / BLOCK_SIZE) * (N / BLOCK_SIZE);
    dim3 grid(num_blocks);
    dim3 block(NUM_THREADS);
    size_t smem = MAX_SHARED_MEMORY;
    
    hipFuncSetAttribute((void*)gemm_mfma_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, smem);
    gemm_mfma_kernel<<<grid, block, smem>>>(gA, gB, gC, M, K, N);
    
    return C;
}
'''

module = load_inline(
    name="gemm_hipkittens",
    cpp_sources=cpp_src,
    cuda_sources=hip_src,
    functions=["matmul_forward"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "-std=c++20",
                       "-I/root/agent/HipKittens/include",
                       "-I/opt/rocm/include/hip",
                       "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
                       "--offload-arch=gfx950"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, A, B):
        return module.matmul_forward(A, B)

