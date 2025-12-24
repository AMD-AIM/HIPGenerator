# HipGenerator

Generate and evaluate HipKittens/HIP kernels on **KernelBench** using an LLM.

## Quick Start

```bash
cd /root/HipGenerator

# Set required environment variable
export LLM_GATEWAY_KEY='your_key_here'

# Test Level1 problems 1-10
./run_batch.sh 1 10

# Test Level2 specific problem
./run_batch.sh --level2 --problem "76_Gemm_Add_ReLU"

# Test all Level1 problems
./run_batch.sh 1 40 --level1
```

## Unified Test Entry: `run_batch.sh`

**All testing should be done through `run_batch.sh`** - this is the single entry point for all tests.

### Usage

```bash
./run_batch.sh [start_id] [end_id] [options]

Options:
  --level1         Test KernelBench level1 (default)
  --level2         Test KernelBench level2
  --problem NAME   Test specific problem by name
  --help           Show help
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_GATEWAY_KEY` | **Yes** | - | API key for LLM gateway |
| `NUM_SAMPLES` | No | 3 | Parallel samples per attempt |
| `MAX_ATTEMPTS` | No | 3 | Max retry attempts |
| `PYTORCH_ROCM_ARCH` | No | gfx950 | GPU architecture |
| `DATASET_BASE` | No | /root/agent/kernel-agent/datasets/KernelBench | Dataset root |

### Examples

```bash
# Test Level1 problems 1-10 with 3 samples, 3 attempts
export LLM_GATEWAY_KEY='...'
./run_batch.sh 1 10

# Quick test with 1 sample, 1 attempt
NUM_SAMPLES=1 MAX_ATTEMPTS=1 ./run_batch.sh --level2 --problem "76_Gemm"

# Full Level1 test
./run_batch.sh 1 40 --level1

# Test all Level2 GEMM problems
./run_batch.sh --level2 --problem "Gemm"
```

## Results

```
results/
├── summary.json            # Overall summary (JSON)
├── detailed_report.txt     # Human-readable report
├── detailed_report.json    # Detailed JSON report
├── batch_run.log           # Execution log
└── <problem_name>/         # Per-problem results
    ├── code_1.py           # Generated code
    ├── result_1.json       # Evaluation result
    ├── best_code.py        # Best performing code
    └── best_result.json    # Best result
```

### Result Status

- **✓ Success**: Accuracy pass + speedup >= 1.0x
- **⚠ Partial**: Accuracy pass + speedup < 1.0x
- **✗ Failed**: Accuracy failed or compile error

## Prompt Configuration

Prompts are auto-selected via `prompts/config.json`:

```json
{
  "default_prompt": "hipkittens_gemm_v3.txt",
  "patterns": {
    "matmul|gemm": "hipkittens_gemm_v3.txt",
    "relu|sigmoid": "elementwise_bf16.txt"
  }
}
```

### Key Prompts

| Prompt | Use Case |
|--------|----------|
| `hipkittens_gemm_v3.txt` | GEMM/Matmul (main, optimized) |
| `elementwise_bf16.txt` | Element-wise ops (ReLU, etc.) |

## Critical Rules (Enforced in Prompts)

1. **禁止 PyTorch 矩阵乘法**: `torch.mm`, `torch.matmul`, `torch.bmm`, `F.linear` 全部禁止
2. **nn.Linear 处理**: 使用 `weight.contiguous()` (不转置)，`mma_ABt(x, weight)` = x @ weight.T
3. **BF16 转换**: 使用 `hip_bfloat16(float_val)` 和 `static_cast<float>(bf16_val)`

## Project Structure

```
/root/HipGenerator/
├── run_batch.sh          # ⭐ Unified test entry point
├── generate.py           # LLM code generator
├── eval.py               # Evaluator with profiling
├── generate_report.py    # Report generator
├── monitor.sh            # Progress monitor
├── prompts/              # Prompt templates
│   ├── config.json       # Auto-selection rules
│   └── *.txt             # Prompt files
├── docs/                 # Documentation
└── results/              # Test outputs (gitignored)
```

## Monitoring Progress

```bash
# Watch live progress
./monitor.sh

# Or use tail
tail -f results/batch_run.log
```

## Generating Reports

Reports are auto-generated after `run_batch.sh` completes:

```bash
# Manual report generation
python3 generate_report.py results/

# View report
cat results/detailed_report.txt
```

## Current Status (Dec 2025)

- **Level2 Problem 76 (Gemm_Add_ReLU)**: Accuracy ✓, Speedup 0.83-0.85x
- **Square matrices (4096², 8192²)**: 1.10x speedup verified
- **Non-square matrices**: ~0.8-0.9x (optimization in progress)

## Notes

- GEMM kernels use HipKittens MFMA instructions (gfx950)
- Always use `run_batch.sh` for consistent testing
- Results are in `results/` (gitignored)
- See `docs/GEMM_OPTIMIZATION_STATUS.md` for optimization details
