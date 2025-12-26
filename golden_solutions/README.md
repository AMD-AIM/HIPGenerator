# Triton Golden Solutions for MI350 (gfx950)

10 个典型 GEMM/Matmul 问题的高性能 Triton 实现，每个都包含原始 Model 和优化后的 ModelNew。

## 硬件配置
- **GPU**: AMD MI350 (gfx950)
- **XCDs**: 32
- **CUs**: 256
- **LDS per CU**: 160 KB

## 优化策略
1. **XCD Swizzle**: 32 XCDs 工作负载均衡分发
2. **L2 Cache Grouping**: GROUP_M=8 提升 L2 缓存命中率
3. **Pingpong Scheduling**: `TRITON_HIP_USE_BLOCK_PINGPONG=1`
4. **Software Pipelining**: `num_stages=3` 对矩形矩阵更优
5. **Float32 累加器**: 保证精度
6. **Kernel Fusion**: 融合 GEMM + Activation
7. **Single-pass Softmax**: 整行加载避免多次遍历

## 性能结果 (最新)

| # | 问题 | Shape | 类型 | Speedup |
|---|------|-------|------|---------|
| 01 | square_gemm | 4096x4096 | 方阵 GEMM | **1.01x** ✅ |
| 04 | gemm_bias_relu | 1024x8192 | 融合 GEMM | 0.98x ⭐ |
| 10 | gemm_gelu_softmax | 1024x4096 | 融合+Softmax | **0.96x** ⭐ |
| 09 | gemm_sigmoid_sum | 1024x4096 | 融合+规约 | **0.95x** ⭐ |
| 05 | gemm_divide_gelu | 1024x8192 | 融合 GEMM | 0.88x ✓ |
| 07 | gemm_swish_scaling | 1024x4096 | 融合 GEMM | 0.85x ✓ |
| 08 | rectangular_gemm | 1024x4096x2048 | 矩形 GEMM | 0.84x ✓ |
| 06 | tall_skinny | 16384x16x1024 | 瘦高矩阵 | 0.81x ✓ |
| 02 | batched_gemm | 128x512x1024x2048 | BMM | 0.80x ✓ |
| 03 | transposed_A | 2048x8192x4096 | A.T @ B | 0.71x |

**✅ = 超越 rocBLAS, ⭐ = >= 0.9x, ✓ = >= 0.8x**

**统计: 1/10 超越 rocBLAS, 4/10 >= 0.9x, 9/10 >= 0.8x**

## 关键优化发现

### 1. num_stages=3 对矩形矩阵更优
对于 1024x4096x4096 形状:
- `num_stages=2`: 0.72x rocBLAS
- `num_stages=3`: **0.90x rocBLAS** (+25% 提升)

### 2. Single-pass Softmax
对于 N=4096 的 Softmax:
- Two-pass (BLOCK_N=256): 0.35x rocBLAS
- Single-pass (BLOCK_N=4096): **0.72x rocBLAS** (+2x 提升)

### 3. 09/10 从 ~0.75x 提升到 ~0.95x
通过以上两个优化，09_gemm_sigmoid_sum 和 10_gemm_gelu_softmax 性能大幅提升。

## 环境变量

```bash
export TRITON_HIP_USE_BLOCK_PINGPONG=1  # 启用 pingpong 调度
export TRITON_HIP_USE_ASYNC_COPY=1       # 启用异步拷贝
```

## 文件结构

每个文件包含:
- `class Model`: 原始 PyTorch 实现 (参考)
- `class ModelNew`: 优化的 Triton 实现
- `get_inputs()`: 测试输入
- `get_init_inputs()`: 模型初始化参数
- 验证和性能测试代码

## 使用示例

```python
from golden_solutions.s01_square_gemm import Model, ModelNew, get_inputs

# 原始模型
ref_model = Model().cuda()

# 优化模型
new_model = ModelNew().cuda()

# 测试
A, B = get_inputs()
A, B = A.cuda(), B.cuda()

ref_out = ref_model(A, B)  # rocBLAS
new_out = new_model(A, B)  # Triton
```

## Triton 源码修改

位于 `/root/triton_dev`，添加了环境变量控制：

```bash
# 可选的 lgkmcnt 控制 (测试显示默认更好)
export TRITON_AMD_FINE_LGKMCNT=4  # 允许 4 个未完成的 LDS 操作
```

修改文件:
- `third_party/amd/lib/TritonAMDGPUToLLVM/MemoryOpToLLVM.cpp`
- `third_party/amd/lib/TritonAMDGPUTransforms/BlockPingpong.cpp`

## 性能瓶颈分析

### 03_transposed_A (0.71x)
- rocBLAS 使用 LDS 进行转置，避免非连续内存访问
- Triton 当前实现直接读取转置数据，效率较低

### 剩余差距来源
1. **指令效率**: Triton 每元素指令数比 rocBLAS 多 ~55%
2. **Kernel Launch**: 多 kernel 方案有额外开销
3. **Tensile 专门优化**: rocBLAS 针对每个 shape 有专门 tune 的 kernel
