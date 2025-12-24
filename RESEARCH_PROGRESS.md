# HipKittens GEMM Kernel Generation Research Progress

## Date: Dec 24, 2025

## Current Status

Successfully developed a prompt that enables LLMs to generate correct HipKittens GEMM kernels with:
- **Adaptive tile sizing**: 256x256 for large matrices, 128x128 for smaller ones
- **Shared memory with double buffering**
- **Instruction scheduling optimizations**

## Test Results (Latest)

### Level1 GEMM Problems
| Problem | Speedup | Status |
|---------|---------|--------|
| 1_Square_matrix_multiplication_ | **0.85x** | ✓ PASS |
| 2_Standard_matrix_multiplication_ | **0.66x** | ✓ PASS |
| 6_Matmul_with_large_K_dimension_ | 0.00x | ✗ Failed |
| 7_Matmul_with_small_K_dimension_ | **0.68x** | ✓ PASS |
| 8_Matmul_with_irregular_shapes_ | **0.91x** | ✓ PASS |
| 9_Tall_skinny_matrix_multiplication_ | **0.54x** | ✓ PASS |
| 16_Matmul_with_transposed_A | **0.52x** | ✓ PASS |
| 17_Matmul_with_transposed_B | **0.57x** | ✓ PASS |

### Level2 GEMM Problems
| Problem | Speedup | Status |
|---------|---------|--------|
| 12_Gemm_Multiply_LeakyReLU | **0.50x** | ✓ PASS |
| 29_Matmul_Mish_Mish | 0.00x | ✗ Failed |
| 40_Matmul_Scaling_ResidualAdd | 0.00x | ✗ Failed |
| 76_Gemm_Add_ReLU | **0.62x** | ✓ PASS |
| 86_Matmul_Divide_GELU | **0.49x** | ✓ PASS |

**Summary**: 10/13 tests pass (77%)

## Performance Evolution

| Version | Description | Performance |
|---------|-------------|-------------|
| Baseline | Global-to-register | 0.16x |
| + Shared memory | Double buffering | 0.40x |
| + sched_barrier | Instruction scheduling | 0.60x |
| + Bt caching | Avoid redundant transpose | 0.85x |
| + Adaptive tiles | 256/128 block selection | 0.91x peak |

## Key Techniques

### 1. Adaptive Block Size Selection
```python
if min(M, N) >= 512:
    BLOCK_SIZE = 256  # Large matrices
else:
    BLOCK_SIZE = 128  # Smaller matrices
```

### 2. Double Buffering with Shared Memory
```cpp
extern __shared__ alignment_dummy __shm[];
shared_allocator al((int*)&__shm[0]);
ST_A (&As)[2][2] = al.allocate<ST_A, 2, 2>();  // 2 buffers x 2 halves
```

### 3. Instruction Scheduling
```cpp
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(1);
mma_ABt(c_accum, a_reg, b_reg, c_accum);
__builtin_amdgcn_s_setprio(0);
```

### 4. Synchronization Pattern
```cpp
// After global loads
asm volatile("s_waitcnt vmcnt(0)\n");
__builtin_amdgcn_s_barrier();

// After shared->register loads
asm volatile("s_waitcnt lgkmcnt(0)\n");
```

## Files

- **Main prompt**: `prompts/hipkittens_gemm.txt`
- **Config**: `prompts/config.json`
- **Reference kernel**: `test_gemm_reference.py`
- **Evaluation**: `eval.py`
- **Generation loop**: `run_loop.py`

## Known Issues

1. **Large K dimension** (K > 4096): Some accuracy issues
2. **Level2 nn.Linear problems**: Some produce NaN due to weight initialization handling
3. **Batched GEMM**: Not supported (uses base prompt)

## Next Steps

1. Fix remaining Level2 issues (nn.Linear weight handling)
2. Improve large K dimension handling
3. Consider deeper pipelining for higher performance
