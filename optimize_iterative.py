#!/usr/bin/env python3
"""
Iterative optimization framework for GEMM kernel generation.
Guides LLM through progressive optimization levels:
1. Base: Simple shared memory double buffering
2. Level 1: Fine-grained waitcnt + sched_barrier
3. Level 2: srsrc_base G::load + readfirstlane hoisting
4. Level 3: 8c4p producer-consumer pattern
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

# Optimization levels
OPTIMIZATION_LEVELS = {
    "base": {
        "name": "Base Double Buffering",
        "prompt": "hipkittens_gemm.txt",
        "description": "Simple shared memory with double buffering",
        "target_speedup": 0.5
    },
    "level1": {
        "name": "Fine-grained Scheduling", 
        "prompt": "hipkittens_gemm_opt1.txt",
        "description": "Add sched_barrier + fine-grained waitcnt",
        "target_speedup": 0.7
    },
    "level2": {
        "name": "Memory Optimizations",
        "prompt": "hipkittens_gemm_opt2.txt", 
        "description": "Add srsrc_base G::load + readfirstlane hoisting",
        "target_speedup": 0.85
    },
    "level3": {
        "name": "Producer-Consumer",
        "prompt": "hipkittens_gemm_opt3.txt",
        "description": "8c4p producer-consumer pattern",
        "target_speedup": 0.95
    }
}

def run_eval(code_path: str, problem_path: str) -> dict:
    """Run evaluation and return result dict."""
    import subprocess
    result = subprocess.run(
        ["python", "eval.py", "--code", code_path, "--problem", problem_path],
        capture_output=True, text=True, cwd="/root/HipGenerator"
    )
    
    # Parse output
    output = result.stdout + result.stderr
    result_dict = {
        "compile_success": "Compile: ✓" in output,
        "accuracy_pass": "Accuracy: ✓" in output or "Accuracy: PASS" in output,
        "speedup": 0.0,
        "error": None
    }
    
    # Extract speedup
    for line in output.split('\n'):
        if "Speedup:" in line:
            try:
                result_dict["speedup"] = float(line.split("Speedup:")[1].strip().rstrip('x'))
            except:
                pass
        if "NaN" in line or "FAIL" in line:
            result_dict["error"] = line.strip()
    
    return result_dict


def generate_with_level(problem_path: str, level: str, attempt: int = 1) -> str:
    """Generate code at specific optimization level."""
    import subprocess
    
    level_info = OPTIMIZATION_LEVELS[level]
    prompt_file = level_info["prompt"]
    
    output_dir = Path(f"results_opt/{Path(problem_path).stem}/{level}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"code_{attempt}.py"
    
    # Run generate.py with specific prompt
    result = subprocess.run(
        ["python", "generate.py", 
         "--problem", problem_path,
         "--prompt", f"prompts/{prompt_file}",
         "--output", str(output_path),
         "--samples", "1"],
        capture_output=True, text=True, cwd="/root/HipGenerator"
    )
    
    if output_path.exists():
        return str(output_path)
    return None


def iterative_optimize(problem_path: str, max_attempts: int = 3):
    """
    Iteratively optimize a kernel through multiple levels.
    Start from base, move to next level when target is met.
    """
    print(f"\n{'='*60}")
    print(f"ITERATIVE OPTIMIZATION: {Path(problem_path).stem}")
    print(f"{'='*60}")
    
    best_result = {"level": None, "speedup": 0.0, "code_path": None}
    current_level = "base"
    
    for level, level_info in OPTIMIZATION_LEVELS.items():
        print(f"\n--- Level: {level_info['name']} ---")
        print(f"Description: {level_info['description']}")
        print(f"Target: {level_info['target_speedup']}x")
        
        level_best = {"speedup": 0.0, "code_path": None}
        
        for attempt in range(1, max_attempts + 1):
            print(f"\n  Attempt {attempt}/{max_attempts}")
            
            # Generate code
            code_path = generate_with_level(problem_path, level, attempt)
            if not code_path:
                print(f"    Failed to generate code")
                continue
            
            # Evaluate
            result = run_eval(code_path, problem_path)
            
            if not result["compile_success"]:
                print(f"    Compile: FAILED")
                continue
                
            if not result["accuracy_pass"]:
                print(f"    Accuracy: FAILED ({result.get('error', 'unknown')})")
                continue
            
            speedup = result["speedup"]
            print(f"    Accuracy: PASS, Speedup: {speedup:.2f}x")
            
            if speedup > level_best["speedup"]:
                level_best = {"speedup": speedup, "code_path": code_path}
            
            if speedup >= level_info["target_speedup"]:
                print(f"    Target met! Moving to next level.")
                break
        
        if level_best["speedup"] > best_result["speedup"]:
            best_result = {"level": level, **level_best}
        
        # If we didn't meet target, try optimization hints
        if level_best["speedup"] < level_info["target_speedup"]:
            print(f"  Target not met at {level}. Best: {level_best['speedup']:.2f}x")
            # Could add feedback loop here
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"Best result: {best_result['level']} - {best_result['speedup']:.2f}x")
    print(f"Code: {best_result['code_path']}")
    print(f"{'='*60}")
    
    return best_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, help="Problem file path")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    
    result = iterative_optimize(args.problem, args.max_attempts)
    
    # Save result
    output_file = Path("results_opt") / f"{Path(args.problem).stem}_result.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

