# HipGenerator

LLM-based HipKittens/Triton kernel generator for AMD GPUs, targeting KernelBench GEMM problems.

## Quick Start

```bash
cd /root/HipGenerator

# Set API key
export LLM_GATEWAY_KEY='your_key_here'

# Test a single problem with HipKittens (default)
python run_loop.py --problem /path/to/problem.py --max-attempts 3

# Test a single problem with Triton
python run_loop.py --problem /path/to/problem.py --max-attempts 3 --backend triton

# Run batch GEMM test (HipKittens)
./batch_test_gemm.sh

# Run batch GEMM test (Triton)
./batch_test_triton_gemm.sh
```

## Supported Backends

### HipKittens (default)
- Uses C++/HIP with HipKittens library for AMD CDNA4 GPUs
- Generates `load_inline` code with MFMA instructions
- Requires HipKittens library installed at `/root/agent/HipKittens`

### Triton
- Uses Triton language for high-performance GPU kernels
- Pure Python with `@triton.jit` decorators
- Supports `@triton.autotune` for automatic parameter tuning
- Simpler development workflow (no C++ compilation)

## Current Status (Dec 2025)

### HipKittens Results
Successfully developed prompts that enable LLMs to generate correct HipKittens GEMM kernels with:
- **Adaptive tile sizing**: 256x256 for large matrices, 128x128 for smaller ones
- **Shared memory with double buffering**
- **Instruction scheduling optimizations**

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

### Triton Results (WIP)
Triton backend is under development. Target: exceed rocBLAS performance.

## Project Structure

```
HipGenerator/
├── run_loop.py              # Main generation loop
├── generate.py              # LLM code generator
├── eval.py                  # Evaluator with profiling
├── batch_test_gemm.sh       # Batch test script (HipKittens)
├── batch_test_triton_gemm.sh # Batch test script (Triton)
├── test_gemm_reference.py   # Reference kernel implementation
├── prompts/
│   ├── config.json          # Prompt selection rules
│   ├── hipkittens_gemm.txt  # Main HipKittens GEMM prompt
│   ├── hipkittens_base.txt  # Base HipKittens prompt
│   ├── triton_gemm.txt      # Main Triton GEMM prompt
│   ├── triton_base.txt      # Base Triton prompt
│   └── elementwise_bf16.txt # Element-wise ops prompt
└── results/                 # Test outputs (gitignored)
```

## Usage

### Single Problem Test

```bash
# HipKittens backend (default)
python run_loop.py --problem /path/to/problem.py --max-attempts 3

# Triton backend
python run_loop.py --problem /path/to/problem.py --max-attempts 3 --backend triton
```

### Generate Code Only

```bash
# Generate HipKittens code
python generate.py --problem /path/to/problem.py --output output.py

# Generate Triton code
python generate.py --problem /path/to/problem.py --output output.py --backend triton
```

### Evaluate Generated Code

```bash
# Evaluate HipKittens code
python eval.py --code results/problem_name/code_1.py --problem /path/to/problem.py

# Evaluate Triton code
python eval.py --code results/problem_name/code_1.py --problem /path/to/problem.py --backend triton
```

### Batch GEMM Test

```bash
# HipKittens
./batch_test_gemm.sh

# Triton
./batch_test_triton_gemm.sh

# Or pass backend as argument
./batch_test_triton_gemm.sh triton
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_GATEWAY_KEY` | **Yes** | - | API key for LLM gateway |
| `PYTORCH_ROCM_ARCH` | No | gfx950 | GPU architecture |

## Key Techniques

### HipKittens

#### 1. Adaptive Block Size Selection
```python
if min(M, N) >= 512:
    BLOCK_SIZE = 256  # Large matrices
else:
    BLOCK_SIZE = 128  # Smaller matrices
```

#### 2. Double Buffering with Shared Memory
```cpp
extern __shared__ alignment_dummy __shm[];
shared_allocator al((int*)&__shm[0]);
ST_A (&As)[2][2] = al.allocate<ST_A, 2, 2>();  // 2 buffers x 2 halves
```

#### 3. Instruction Scheduling
```cpp
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(1);
mma_ABt(c_accum, a_reg, b_reg, c_accum);
__builtin_amdgcn_s_setprio(0);
```

### Triton

#### 1. Autotune with Multiple Configs
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
    ...
```

#### 2. Grouped Ordering for L2 Cache
```python
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

#### 3. tl.dot for MFMA
```python
accumulator = tl.dot(a, b, accumulator)  # Maps to AMD MFMA instructions
```

## Prompt Configuration

Prompts are auto-selected via `prompts/config.json`:

```json
{
  "default_prompt": "hipkittens_base.txt",
  "triton_default_prompt": "triton_base.txt",
  "patterns": {
    "matmul|gemm": "hipkittens_gemm.txt"
  },
  "triton_patterns": {
    "matmul|gemm": "triton_gemm.txt"
  }
}
```

## Known Issues

### HipKittens
1. **Large K dimension** (K > 4096): Some accuracy issues
2. **Level2 nn.Linear problems**: Some produce NaN due to weight initialization
3. **Batched GEMM**: Not supported (uses base prompt)

### Triton
1. Backend under development
2. Autotune may take time on first run

## Critical Rules (Enforced in Prompts)

### HipKittens
1. **No PyTorch GEMM**: `torch.mm`, `torch.matmul`, `torch.bmm`, `F.linear` are forbidden
2. **nn.Linear handling**: Use `weight.contiguous()` (no transpose), `mma_ABt(x, weight)` = x @ weight.T
3. **BF16 conversion**: Use `hip_bfloat16(float_val)` and `static_cast<float>(bf16_val)`

### Triton
1. **No PyTorch GEMM**: Use `tl.dot()` for matrix multiplication
2. **Use autotune**: Multiple configurations for optimal performance
3. **Float32 accumulators**: Use `tl.float32` for accumulator precision

## Notes

- GEMM kernels use MFMA instructions (gfx950/CDNA4)
- Results are stored in `results/` or `results_triton/` directory
- See `test_gemm_reference.py` for working kernel example
