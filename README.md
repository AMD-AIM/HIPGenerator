# HipKittens Kernel Generation Auto Test Framework

自动化测试框架，用于批量测试 LLM 生成 HipKittens kernel 的能力。

## 文件说明

| 文件 | 功能 |
|------|------|
| `generate.py` | 调用 LLM 根据 system prompt 生成代码 |
| `eval.py` | 评估生成代码的正确性和性能 |
| `run_batch.sh` | 批量运行测试，自动迭代优化 |
| `generate_report.py` | 生成详细的测试报告 |
| `monitor.sh` | 监控测试进度 |

## 使用方法

## 环境变量

- **`LLM_GATEWAY_KEY`**: 必填。LLM Gateway 的 API key（`run_batch.sh` 会检查该变量是否为空）。
- **`KERNELBENCH_DIR`**: 可选。KernelBench level1 数据集目录。默认：`/root/agent/kernel-agent/datasets/KernelBench/level1`
- **`NUM_SAMPLES`**: 可选。每个 attempt 并行生成多少个 sample（默认 3）。

### 1. 单独生成代码

```bash
python generate.py \
    --problem /path/to/problem.py \
    --output output_code.py \
    [--prompt custom_prompt.txt] \
    [--response-file llm_response.txt]
```

### 2. 单独评估代码

```bash
python eval.py \
    --code generated_code.py \
    --problem /path/to/problem.py \
    [--output result.json]
```

返回码:
- 0: 成功 (正确性通过 + speedup >= 1.0)
- 1: 部分成功 (正确性通过，speedup < 1.0)
- 2: 失败 (正确性未通过)

### 3. 批量测试

```bash
# 设置数据集目录（如有需要）
export KERNELBENCH_DIR=/root/agent/kernel-agent/datasets/KernelBench/level1

# 测试 level1 的任务 1-40
./run_batch.sh 1 40

# 测试特定范围
./run_batch.sh 19 25
```

每个任务最多尝试 3 次，每次失败后会根据错误更新 prompt。

### 4. 监控进度

```bash
./monitor.sh
```

### 5. 生成报告

```bash
python generate_report.py results/
```

## 结果目录结构

```
results/
├── summary.json           # 总体摘要
├── detailed_report.txt    # 详细报告
├── detailed_report.json   # JSON格式报告
├── batch_run.log         # 运行日志
└── <problem_name>/       # 每个问题的目录
    ├── prompt_1.txt      # 第1次尝试的prompt
    ├── code_1.py         # 第1次尝试的代码
    ├── result_1.json     # 第1次尝试的结果
    ├── response_1.txt    # 第1次LLM完整响应
    ├── ...
    ├── best_prompt.txt   # 最佳prompt
    ├── best_code.py      # 最佳代码
    └── best_result.json  # 最佳结果
```

## Prompt 配置系统

框架使用配置文件 `prompts/config.json` 来根据问题类型自动选择合适的 prompt。

### 配置文件结构

```
prompts/
├── config.json              # 配置文件
├── elementwise_bf16.txt     # 元素级操作 (ReLU, Sigmoid 等)
├── gemm_rocblas.txt         # GEMM 使用 rocBLAS
└── gemm_hipkittens.txt      # GEMM 使用 HipKittens MFMA
```

### config.json 格式

```json
{
  "default_prompt": "elementwise_bf16.txt",
  "patterns": {
    "relu|sigmoid|tanh": "elementwise_bf16.txt",
    "matmul|gemm": "gemm_rocblas.txt",
    "attention": "gemm_hipkittens.txt"
  },
  "reflection_hints": {
    "PERFORMANCE TOO SLOW": ["优化建议1", "优化建议2"],
    "Compile error": ["修复建议1", "修复建议2"]
  }
}
```

### 手动指定 Prompt

```bash
python generate.py --problem problem.py --output code.py --prompt prompts/elementwise_bf16.txt
```

## 功能特性

### Core dump 限制
使用 `ulimit -c 0` 禁止生成 core dump 文件，避免 GPU 崩溃时产生大文件。

### rocprof 性能分析
对于性能不达标的 kernel（speedup < 1.0），自动运行 rocprof 进行性能分析：
- 识别真正运行的 kernel 名称
- 检测是否使用了 PyTorch 内置 kernel（如类型转换）
- 将分析结果反馈给 LLM 进行优化

### 反射机制
每次尝试失败后，根据错误类型生成针对性的反馈：
- 编译错误：提供具体的错误信息
- 性能问题：提供 rocprof 分析和优化建议
- GPU 崩溃：提示检查内存对齐和边界

## 当前限制

1. **GEMM/MatMul**: HipKittens 的 MFMA API (st_bf, rt_bf, mma_ABt) 较为复杂，LLM 难以正确生成
2. **简单操作**: ReLU, Sigmoid 等简单操作使用向量化 HIP kernel 可达 0.9x+ baseline
3. **大维度**: rv_naive 仅支持 D <= 2048，需要 tiled 实现

## 性能优化关键

1. **避免类型转换**: 不要使用 `.to(torch::kFloat32)` 和 `.to(x.dtype())`，直接处理原始 dtype
2. **向量化**: 使用 `__half2`, `float4` 等向量类型
3. **Coalesced access**: 保证内存访问连续
4. **使用 __restrict__**: 帮助编译器优化



