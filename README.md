# HipKittens KernelBench Runner

Batch runner for generating and evaluating HipKittens/HIP kernels on **KernelBench (level1)** using an LLM.

## What’s in this repo

| File | Purpose |
|------|---------|
| `generate.py` | Calls the LLM and generates Python code (with `load_inline`) for a KernelBench problem |
| `eval.py` | Compiles/runs the generated code, checks correctness, benchmarks, and optionally runs `rocprofv3` |
| `run_batch.sh` | Batch runner: loops over problems, retries with “reflection” prompts, and keeps best results |
| `generate_report.py` | Generates a readable report from `results/summary.json` |
| `monitor.sh` | Quick progress monitor for a running batch |

## Environment variables

- **`LLM_GATEWAY_KEY`**: **Required.** The LLM gateway API key (validated by `run_batch.sh`).
- **`KERNELBENCH_DIR`**: Optional. Path to the KernelBench level1 directory. Default: `/root/agent/kernel-agent/datasets/KernelBench/level1`
- **`NUM_SAMPLES`**: Optional. Number of parallel samples generated per attempt (default: `3`).

## Usage

### 1) Generate code for a single problem

```bash
python3 generate.py \
  --problem /path/to/problem.py \
  --output output_code.py \
  --prompt prompts/elementwise_bf16.txt \
  --response-file llm_response.txt
```

### 2) Evaluate a single generated file

```bash
python3 eval.py \
  --code output_code.py \
  --problem /path/to/problem.py \
  --output result.json
```

Exit codes:
- `0`: success (accuracy pass + speedup >= 1.0)
- `1`: partial (accuracy pass, speedup < 1.0)
- `2`: failed (accuracy failed)

### 3) Batch run

```bash
export LLM_GATEWAY_KEY='...'

# Override dataset location if needed
export KERNELBENCH_DIR=/root/agent/kernel-agent/datasets/KernelBench/level1

# Run a range (default is 1..40)
./run_batch.sh 1 40

# Run a smaller range
./run_batch.sh 19 25
```

Each problem is tried up to 3 attempts. On failure/slow performance, the runner updates the prompt with error context (“reflection”).

### 4) Monitor progress

```bash
./monitor.sh
```

### 5) Generate a report

```bash
python3 generate_report.py results/
```

## Results directory layout

```
results/
├── summary.json           # overall summary (machine-readable)
├── detailed_report.txt    # detailed report (text)
├── detailed_report.json   # detailed report (json)
├── batch_run.log          # run log
└── <problem_name>/        # one directory per problem
    ├── prompt_1.txt
    ├── code_1.py
    ├── result_1.json
    ├── response_1.txt
    ├── ...
    ├── best_prompt.txt
    ├── best_code.py
    └── best_result.json
```

## Prompt configuration

Prompts are selected automatically via `prompts/config.json` based on the problem name (regex patterns).

Directory:

```
prompts/
├── config.json
├── elementwise_bf16.txt
├── gemm_rocblas.txt
└── gemm_hipkittens.txt
```

Example `config.json`:

```json
{
  "default_prompt": "elementwise_bf16.txt",
  "patterns": {
    "relu|sigmoid|tanh": "elementwise_bf16.txt",
    "matmul|gemm": "gemm_rocblas.txt",
    "attention": "gemm_hipkittens.txt"
  },
  "reflection_hints": {
    "PERFORMANCE TOO SLOW": ["hint1", "hint2"],
    "Compile error": ["hint1", "hint2"]
  }
}
```

## Notes / limitations

- GEMM via HipKittens MFMA APIs can be tricky to generate correctly; prompts/templates exist to guide the model.
- For some workloads, **dtype conversions (`.to(...)`) can dominate runtime**; prompts try to steer away from them.
