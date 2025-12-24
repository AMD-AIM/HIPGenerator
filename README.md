# HipGenerator

LLM-based HipKittens/HIP kernel generator for AMD GPUs, targeting KernelBench GEMM problems.

## Quick Start

```bash
cd /root/HipGenerator

# Set API key
export LLM_GATEWAY_KEY='your_key_here'

# Test a single problem
python run_loop.py --problem /path/to/problem.py --max-attempts 3

# Run batch GEMM test
./batch_test_gemm.sh
```

## Current Status (Dec 2025)

Successfully developed prompts that enable LLMs to generate correct HipKittens GEMM kernels with:
- **Adaptive tile sizing**: 256x256 for large matrices, 128x128 for smaller ones
- **Shared memory with double buffering**
- **Instruction scheduling optimizations**

### Test Results

| Problem | Speedup | Status |
|---------|---------|--------|
| 1_Square_matrix_multiplication_ | **0.85x** | PASS |
| 2_Standard_matrix_multiplication_ | **0.66x** | PASS |
| 6_Matmul_with_large_K_dimension_ | 0.00x | Failed |
| 7_Matmul_with_small_K_dimension_ | **0.68x** | PASS |
| 8_Matmul_with_irregular_shapes_ | **0.91x** | PASS |
| 9_Tall_skinny_matrix_multiplication_ | **0.54x** | PASS |
| 16_Matmul_with_transposed_A | **0.52x** | PASS |
| 17_Matmul_with_transposed_B | **0.57x** | PASS |
| 12_Gemm_Multiply_LeakyReLU | **0.50x** | PASS |
| 29_Matmul_Mish_Mish | 0.00x | Failed |
| 40_Matmul_Scaling_ResidualAdd | 0.00x | Failed |
| 76_Gemm_Add_ReLU | **0.62x** | PASS |
| 86_Matmul_Divide_GELU | **0.49x** | PASS |

**Summary**: 10/13 tests pass (77%), best performance 0.91x

### Performance Evolution

| Version | Description | Performance |
|---------|-------------|-------------|
| Baseline | Global-to-register | 0.16x |
| + Shared memory | Double buffering | 0.40x |
| + sched_barrier | Instruction scheduling | 0.60x |
| + Bt caching | Avoid redundant transpose | 0.85x |
| + Adaptive tiles | 256/128 block selection | 0.91x peak |

## Project Structure

```
HipGenerator/
├── run_loop.py              # Main generation loop
├── generate.py              # LLM code generator
├── eval.py                  # Evaluator with profiling
├── batch_test_gemm.sh       # Batch test script
├── test_gemm_reference.py   # Reference kernel implementation
├── prompts/
│   ├── config.json          # Prompt selection rules
│   ├── hipkittens_gemm.txt  # Main GEMM prompt
│   ├── hipkittens_base.txt  # Base prompt
│   └── elementwise_bf16.txt # Element-wise ops prompt
└── results/                 # Test outputs (gitignored)
```

## Usage

### Single Problem Test

```bash
python run_loop.py --problem /path/to/problem.py --max-attempts 3
```

### Batch GEMM Test

```bash
./batch_test_gemm.sh
```

This tests all GEMM-related problems from KernelBench Level1 and Level2.

### Evaluate Generated Code

```bash
python eval.py --code results/problem_name/code_1.py --problem /path/to/problem.py
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_GATEWAY_KEY` | **Yes** | - | API key for LLM gateway |
| `PYTORCH_ROCM_ARCH` | No | gfx950 | GPU architecture |

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

## Prompt Configuration

Prompts are auto-selected via `prompts/config.json`:

```json
{
  "default_prompt": "hipkittens_base.txt",
  "patterns": {
    "matmul|gemm": "hipkittens_gemm.txt",
    "relu|sigmoid": "elementwise_bf16.txt"
  }
}
```

## Known Issues

1. **Large K dimension** (K > 4096): Some accuracy issues
2. **Level2 nn.Linear problems**: Some produce NaN due to weight initialization
3. **Batched GEMM**: Not supported (uses base prompt)

## Critical Rules (Enforced in Prompts)

1. **No PyTorch GEMM**: `torch.mm`, `torch.matmul`, `torch.bmm`, `F.linear` are forbidden
2. **nn.Linear handling**: Use `weight.contiguous()` (no transpose), `mma_ABt(x, weight)` = x @ weight.T
3. **BF16 conversion**: Use `hip_bfloat16(float_val)` and `static_cast<float>(bf16_val)`

## Notes

- GEMM kernels use HipKittens MFMA instructions (gfx950/CDNA4)
- Results are stored in `results/` directory
- See `test_gemm_reference.py` for working kernel example
