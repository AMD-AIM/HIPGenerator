#!/bin/bash
export PYTORCH_ROCM_ARCH=gfx950

problems=(
    # Level 1 - Basic GEMM
    "/root/agent/kernel-agent/datasets/KernelBench/level1/1_Square_matrix_multiplication_.py"
    "/root/agent/kernel-agent/datasets/KernelBench/level1/2_Standard_matrix_multiplication_.py"
    "/root/agent/kernel-agent/datasets/KernelBench/level1/6_Matmul_with_large_K_dimension_.py"
    "/root/agent/kernel-agent/datasets/KernelBench/level1/7_Matmul_with_small_K_dimension_.py"
    "/root/agent/kernel-agent/datasets/KernelBench/level1/8_Matmul_with_irregular_shapes_.py"
    "/root/agent/kernel-agent/datasets/KernelBench/level1/9_Tall_skinny_matrix_multiplication_.py"
    "/root/agent/kernel-agent/datasets/KernelBench/level1/16_Matmul_with_transposed_A.py"
    "/root/agent/kernel-agent/datasets/KernelBench/level1/17_Matmul_with_transposed_B.py"
    # Level 2 - Fused GEMM
    "/root/agent/kernel-agent/datasets/KernelBench/level2/12_Gemm_Multiply_LeakyReLU.py"
    "/root/agent/kernel-agent/datasets/KernelBench/level2/29_Matmul_Mish_Mish.py"
    "/root/agent/kernel-agent/datasets/KernelBench/level2/40_Matmul_Scaling_ResidualAdd.py"
    "/root/agent/kernel-agent/datasets/KernelBench/level2/76_Gemm_Add_ReLU.py"
    "/root/agent/kernel-agent/datasets/KernelBench/level2/86_Matmul_Divide_GELU.py"
)

echo "================================================"
echo "BATCH GEMM TEST - $(date)"
echo "================================================"

total=0
passed=0
speedups=()

for problem in "${problems[@]}"; do
    name=$(basename "$problem" .py)
    echo -e "\n--- Testing: $name ---"
    
    # Run single attempt
    result=$(python run_loop.py --problem "$problem" --max-attempts 1 2>&1)
    
    # Extract status
    if echo "$result" | grep -q "Accuracy: PASS"; then
        speedup=$(echo "$result" | grep "Best speedup:" | awk '{print $3}')
        echo "✓ PASS - Speedup: $speedup"
        passed=$((passed + 1))
        speedups+=("$speedup")
    else
        error=$(echo "$result" | grep -E "NaN|FAIL|Compile error" | head -1)
        echo "✗ FAIL - $error"
        speedups+=("0.00x")
    fi
    total=$((total + 1))
done

echo ""
echo "================================================"
echo "BATCH SUMMARY"
echo "================================================"
echo "Passed: $passed / $total"
echo ""
echo "Individual results:"
for i in "${!problems[@]}"; do
    name=$(basename "${problems[$i]}" .py)
    echo "  $name: ${speedups[$i]}"
done
