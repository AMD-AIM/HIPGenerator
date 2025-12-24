#!/bin/bash
# Batch test script for Triton GEMM kernel generation
# Tests all GEMM-related problems from KernelBench Level1 and Level2

export PYTORCH_ROCM_ARCH=gfx950

# Check for backend argument (default: triton)
BACKEND="${1:-triton}"
echo "Testing with backend: $BACKEND"

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
echo "BATCH ${BACKEND^^} GEMM TEST - $(date)"
echo "================================================"

total=0
passed=0
speedups=()

# Create results directory for this backend
OUTPUT_DIR="results_${BACKEND}"

for problem in "${problems[@]}"; do
    name=$(basename "$problem" .py)
    echo -e "\n--- Testing: $name ---"
    
    # Run single attempt with specified backend
    result=$(python run_loop.py --problem "$problem" --max-attempts 1 --backend "$BACKEND" --output "$OUTPUT_DIR" 2>&1)
    
    # Extract status
    if echo "$result" | grep -q "Accuracy: PASS"; then
        speedup=$(echo "$result" | grep "Best speedup:" | awk '{print $3}')
        echo "✓ PASS - Speedup: $speedup"
        passed=$((passed + 1))
        speedups+=("$speedup")
    else
        error=$(echo "$result" | grep -E "NaN|FAIL|Compile error|error:" | head -1)
        echo "✗ FAIL - $error"
        speedups+=("0.00x")
    fi
    total=$((total + 1))
done

echo ""
echo "================================================"
echo "BATCH SUMMARY (${BACKEND^^})"
echo "================================================"
echo "Passed: $passed / $total"
echo ""
echo "Individual results:"
for i in "${!problems[@]}"; do
    name=$(basename "${problems[$i]}" .py)
    echo "  $name: ${speedups[$i]}"
done

# Calculate average speedup for passing tests
if [ $passed -gt 0 ]; then
    sum=0
    for s in "${speedups[@]}"; do
        # Extract numeric value (remove 'x' suffix)
        val=$(echo "$s" | sed 's/x$//')
        if [ "$val" != "0.00" ]; then
            sum=$(echo "$sum + $val" | bc)
        fi
    done
    avg=$(echo "scale=2; $sum / $passed" | bc)
    echo ""
    echo "Average speedup (passing tests): ${avg}x"
fi

