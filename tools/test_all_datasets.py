#!/usr/bin/env python3
"""
Test all datasets and categorize results.
Categories:
- GEMM: matrix multiplication, batched mm, transposed, fused operations
- ElementWise: relu, sigmoid, tanh, gelu, swish, etc.
- Norm: layernorm, batchnorm, groupnorm, rmsnorm
- Reduction: sum, mean, max, pooling
- Attention: scaled dot product attention
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).parent.parent

# Problem categorization
CATEGORIES = {
    "GEMM": [
        "1_Square_matrix_multiplication_",
        "2_Standard_matrix_multiplication_",
        "3_Batched_matrix_multiplication",
        "4_Matrix_vector_multiplication_",
        "5_Matrix_scalar_multiplication",
        "8_Matmul_with_irregular_shapes_",
        "9_Tall_skinny_matrix_multiplication_",
        "16_Matmul_with_transposed_A",
        "17_Matmul_with_transposed_B",
        "59_Matmul_Swish_Scaling",
        "76_Gemm_Add_ReLU",
        "99_Matmul_GELU_Softmax",
    ],
    "ElementWise": [
        "19_ReLU",
        "22_Tanh",
        "25_Swish",
        "26_GELU_",
        "88_MinGPTNewGelu",
    ],
    "Norm": [
        "33_BatchNorm",
        "35_GroupNorm_",
        "36_RMSNorm_",
        "40_LayerNorm",
    ],
    "Reduction": [
        "23_Softmax",
        "24_LogSoftmax",
        "41_Max_Pooling_1D",
        "42_Max_Pooling_2D",
        "44_Average_Pooling_1D",
        "45_Average_Pooling_2D",
        "47_Sum_reduction_over_a_dimension",
        "48_Mean_reduction_over_a_dimension",
    ],
    "Attention": [
        "97_ScaledDotProductAttention",
    ],
}

# Target speedups by category
TARGET_SPEEDUPS = {
    "GEMM": 0.8,  # Target 0.8x vs rocBLAS (very competitive)
    "ElementWise": 1.0,  # Target 1.0x (match or beat PyTorch)
    "Norm": 0.8,  # Target 0.8x
    "Reduction": 0.8,  # Target 0.8x
    "Attention": 0.8,  # Target 0.8x
}


def get_category(problem_name: str) -> str:
    """Get the category of a problem."""
    for cat, problems in CATEGORIES.items():
        if problem_name in problems:
            return cat
    return "Unknown"


def run_test(problem_path: str, max_attempts: int = 3) -> dict:
    """Run generation test for a single problem."""
    env = {
        **os.environ,
        "LLM_GATEWAY_KEY": os.environ.get("LLM_GATEWAY_KEY", ""),
        "PYTORCH_ROCM_ARCH": "gfx950"
    }
    
    problem_name = Path(problem_path).stem
    output_dir = SCRIPT_DIR / "results_all" / problem_name
    
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_loop.py"),
        "--problem", problem_path,
        "--output", str(output_dir.parent),
        "--max-attempts", str(max_attempts),
        "--samples", "1",
        "--target-speedup", "0.5",  # Low threshold, measure actual
        "--backend", "triton"
    ]
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        
        best_result_file = output_dir / "best_result.json"
        if best_result_file.exists():
            with open(best_result_file) as f:
                result = json.load(f)
            return {
                "success": result.get("accuracy_pass", False),
                "speedup": result.get("speedup", 0),
                "compiled": result.get("compile_success", False),
                "correct": result.get("accuracy_pass", False),
                "attempts": result.get("attempt", 0),
            }
        return {"success": False, "speedup": 0, "error": "No result file"}
    except subprocess.TimeoutExpired:
        return {"success": False, "speedup": 0, "error": "Timeout"}
    except Exception as e:
        return {"success": False, "speedup": 0, "error": str(e)}


def main():
    if not os.environ.get("LLM_GATEWAY_KEY"):
        print("Error: LLM_GATEWAY_KEY not set")
        sys.exit(1)
    
    datasets_dir = SCRIPT_DIR / "datasets"
    problems = sorted(datasets_dir.glob("*.py"))
    
    results = {}
    category_stats = {cat: {"total": 0, "pass": 0, "speedups": []} for cat in CATEGORIES}
    category_stats["Unknown"] = {"total": 0, "pass": 0, "speedups": []}
    
    print(f"\n{'='*70}")
    print("TESTING ALL DATASETS")
    print(f"{'='*70}\n")
    
    for prob_path in problems:
        prob_name = prob_path.stem
        category = get_category(prob_name)
        target = TARGET_SPEEDUPS.get(category, 0.5)
        
        print(f"[{category}] {prob_name}... ", end="", flush=True)
        
        result = run_test(str(prob_path), max_attempts=3)
        results[prob_name] = {**result, "category": category}
        
        category_stats[category]["total"] += 1
        
        if result.get("correct"):
            speedup = result.get("speedup", 0)
            category_stats[category]["speedups"].append(speedup)
            
            if speedup >= target:
                category_stats[category]["pass"] += 1
                print(f"✓ {speedup:.2f}x (target: {target}x)")
            else:
                print(f"⚠ {speedup:.2f}x < {target}x target")
        else:
            print(f"✗ FAILED ({result.get('error', 'accuracy fail')[:30]})")
    
    # Print summary by category
    print(f"\n{'='*70}")
    print("RESULTS BY CATEGORY")
    print(f"{'='*70}\n")
    
    overall_pass = 0
    overall_total = 0
    
    for cat in list(CATEGORIES.keys()) + ["Unknown"]:
        stats = category_stats[cat]
        if stats["total"] == 0:
            continue
        
        avg_speedup = sum(stats["speedups"]) / len(stats["speedups"]) if stats["speedups"] else 0
        pass_rate = stats["pass"] / stats["total"] * 100 if stats["total"] > 0 else 0
        
        print(f"\n{cat}:")
        print(f"  Total: {stats['total']}")
        print(f"  Accuracy Pass: {len(stats['speedups'])}/{stats['total']}")
        print(f"  Target Pass: {stats['pass']}/{stats['total']} ({pass_rate:.1f}%)")
        print(f"  Avg Speedup: {avg_speedup:.2f}x")
        
        overall_pass += stats["pass"]
        overall_total += stats["total"]
    
    print(f"\n{'='*70}")
    print(f"OVERALL: {overall_pass}/{overall_total} ({100*overall_pass/overall_total:.1f}%) meet target")
    print(f"{'='*70}")
    
    # List problems that need improvement
    print(f"\n{'='*70}")
    print("PROBLEMS NEEDING IMPROVEMENT")
    print(f"{'='*70}\n")
    
    for prob_name, result in results.items():
        category = result.get("category", "Unknown")
        target = TARGET_SPEEDUPS.get(category, 0.5)
        speedup = result.get("speedup", 0)
        
        if not result.get("correct"):
            print(f"  ✗ {prob_name} [{category}]: ACCURACY FAIL")
        elif speedup < target:
            print(f"  ⚠ {prob_name} [{category}]: {speedup:.2f}x < {target}x")
    
    # Save results
    output_file = SCRIPT_DIR / "all_datasets_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "results": results,
            "category_stats": {
                k: {**v, "speedups": v["speedups"]} 
                for k, v in category_stats.items()
            }
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

