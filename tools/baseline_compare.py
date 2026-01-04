#!/usr/bin/env python3
"""
Compare HIPGenerator against the baseline TritonAgent API.

This script calls the TritonAgent baseline API (https://kernelagent.oneclickamd.ai/)
and compares results with our local generation.

Usage:
    python tools/baseline_compare.py --problem datasets/19_ReLU.py
    python tools/baseline_compare.py --problem datasets/25_Swish.py --prompt high_performance
    python tools/baseline_compare.py --baseline-only datasets/19_ReLU.py  # Only call baseline
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from gradio_client import Client
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    print("Warning: gradio_client not installed. Run: pip install gradio_client")

# Baseline API endpoint
BASELINE_URL = "https://kernelagent.oneclickamd.ai/"

# Prompt templates matching baseline
PROMPTS = {
    "high_correctness": """
### **CRITICAL: CORRECTNESS FIRST, PERFORMANCE SECOND** ###
You MUST guarantee the correctness of ModelNew. Do NOT cheat by simplifying logic or skipping computations.

### **Common Errors to AVOID** ###

**ERROR 1: Numerical Accuracy Issues (GELU, Normalization, Softmax)**
- ❌ WRONG: Using fp16 directly for `tl.exp()`, `tl.log()`, or complex math operations
- ✅ CORRECT: Always cast to fp32 for intermediate calculations, then cast back.

**ERROR 2: Missing @triton.jit Decorator**
- ❌ WRONG: Helper functions without decorator that use Triton operations
- ✅ CORRECT: Add `@triton.jit` to ALL functions using Triton ops

**ERROR 3: Invalid Triton APIs**
- ❌ FORBIDDEN APIs (do NOT use): tl.math.tanh, tl.tanh, tl.astype, tl.floor_div

**ERROR 4: Control Flow Restrictions**
- ❌ FORBIDDEN: `continue` and `break` statements in Triton kernels
- ✅ CORRECT: Use tl.where() for conditional execution

### **Priority: CORRECTNESS > PERFORMANCE** ###
If unsure, choose the more numerically stable and correct approach.
""",
    
    "high_performance": """
### **HIGH PERFORMANCE OPTIMIZATION MODE** ###
Target: >2x speedup over PyTorch reference

### Aggressive Optimizations ###
1. **Use LARGE block sizes**: 8192, 16384, 32768 with high num_warps (8-16)
2. **Use @triton.autotune with many configurations and num_stages**
3. **Use fast math approximations where acceptable**
4. **Maximize memory coalescing and bandwidth utilization**

### Performance Configuration Example ###
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
```

### Priority: PERFORMANCE > ABSOLUTE CORRECTNESS ###
Small numerical differences (1e-3 relative tolerance) are acceptable for >2x speedup.
"""
}


def call_baseline_api(ref_code: str, prompt_type: str = "high_correctness", 
                      max_retries: int = 2, target_speedup: float = 1.0) -> dict:
    """Call the baseline TritonAgent API to generate a kernel.
    
    Args:
        ref_code: Reference PyTorch implementation
        prompt_type: "high_correctness" or "high_performance"
        max_retries: Max retry attempts
        target_speedup: Target speedup threshold
        
    Returns:
        dict with generated code and evaluation results
    """
    if not HAS_GRADIO:
        return {"success": False, "error": "gradio_client not installed"}
    
    try:
        client = Client(BASELINE_URL)
        
        custom_prompt = PROMPTS.get(prompt_type, PROMPTS["high_correctness"])
        
        # Call the submit_generation endpoint
        result = client.predict(
            ref_arch_src=ref_code,
            backend="triton",
            server_type="anthropic",
            model_name="claude-opus-4.5",  # Correct model name
            gpu_arch="CDNA",
            max_tokens=4096,
            temperature=1.0,
            custom_prompt=custom_prompt,
            max_retries=max_retries,
            target_speedup=target_speedup,
            api_name="/submit_generation"
        )
        
        # Parse result
        generated_code = result[0] if len(result) > 0 else None
        eval_results = result[1] if len(result) > 1 else None
        status = result[2] if len(result) > 2 else None
        
        # Extract metrics from eval_results
        speedup = 0.0
        compiled = False
        correct = False
        
        if eval_results:
            # Parse speedup
            speedup_match = re.search(r'\*\*(\d+\.?\d*)x\*\*', eval_results)
            if speedup_match:
                speedup = float(speedup_match.group(1))
            
            compiled = "✅ Compiled" in eval_results
            correct = "✅ Correct" in eval_results
        
        return {
            "success": generated_code is not None and len(generated_code) > 100,
            "code": generated_code,
            "eval_results": eval_results,
            "status": status,
            "speedup": speedup,
            "compiled": compiled,
            "correct": correct,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "code": None,
            "eval_results": None,
            "speedup": 0.0,
            "compiled": False,
            "correct": False,
            "error": str(e)
        }


def run_local_generation(problem_path: str, max_attempts: int = 2) -> dict:
    """Run local HIPGenerator on a problem."""
    import subprocess
    import tempfile
    
    script_dir = Path(__file__).parent.parent
    problem_name = Path(problem_path).stem
    output_dir = script_dir / "results_compare" / problem_name
    
    cmd = [
        sys.executable, str(script_dir / "run_loop.py"),
        "--problem", problem_path,
        "--output", str(output_dir.parent),
        "--max-attempts", str(max_attempts),
        "--samples", "1",
        "--target-speedup", "0.5",
        "--backend", "triton"
    ]
    
    env = {**os.environ, 
           "LLM_GATEWAY_KEY": os.environ.get("LLM_GATEWAY_KEY", ""),
           "PYTORCH_ROCM_ARCH": "gfx950"}
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        
        # Load results
        best_result_file = output_dir / "best_result.json"
        best_code_file = output_dir / "best_code.py"
        
        if best_result_file.exists():
            with open(best_result_file) as f:
                local_eval = json.load(f)
            
            local_code = ""
            if best_code_file.exists():
                with open(best_code_file) as f:
                    local_code = f.read()
            
            return {
                "success": local_eval.get("accuracy_pass", False),
                "code": local_code,
                "speedup": local_eval.get("speedup", 0),
                "compiled": local_eval.get("compile_success", False),
                "correct": local_eval.get("accuracy_pass", False),
                "error": None
            }
        else:
            return {
                "success": False,
                "code": None,
                "speedup": 0,
                "compiled": False,
                "correct": False,
                "error": "No result file generated"
            }
    except Exception as e:
        return {
            "success": False,
            "code": None,
            "speedup": 0,
            "compiled": False,
            "correct": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Compare HIPGenerator with baseline API")
    parser.add_argument("--problem", help="Path to problem file")
    parser.add_argument("--prompt", choices=["high_correctness", "high_performance"],
                       default="high_performance", help="Prompt type for baseline")
    parser.add_argument("--baseline-only", action="store_true", 
                       help="Only call baseline, don't run local")
    parser.add_argument("--local-only", action="store_true",
                       help="Only run local, don't call baseline")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries")
    parser.add_argument("--target-speedup", type=float, default=1.5, help="Target speedup")
    parser.add_argument("--save-code", action="store_true", help="Save generated code")
    args = parser.parse_args()
    
    if not args.problem:
        print("Error: --problem is required")
        sys.exit(1)
    
    problem_path = Path(args.problem)
    if not problem_path.exists():
        print(f"Error: {problem_path} not found")
        sys.exit(1)
    
    problem_name = problem_path.stem
    
    # Load reference code
    with open(problem_path) as f:
        ref_code = f.read()
    
    print(f"\n{'='*60}")
    print(f"COMPARING: {problem_name}")
    print(f"Prompt: {args.prompt}")
    print(f"{'='*60}")
    
    baseline_result = None
    local_result = None
    
    # Call baseline
    if not args.local_only:
        print("\n--- Calling Baseline API (TritonAgent) ---")
        baseline_result = call_baseline_api(
            ref_code, 
            prompt_type=args.prompt,
            max_retries=args.max_retries,
            target_speedup=args.target_speedup
        )
        
        if baseline_result["success"]:
            status = "✓" if baseline_result["correct"] else "✗"
            print(f"  Result: {status} Compiled={baseline_result['compiled']}, "
                  f"Correct={baseline_result['correct']}, Speedup={baseline_result['speedup']:.2f}x")
            
            if args.save_code and baseline_result["code"]:
                output_file = f"baseline_{problem_name}_{args.prompt}.py"
                with open(output_file, 'w') as f:
                    f.write(baseline_result["code"])
                print(f"  Saved to: {output_file}")
        else:
            print(f"  Failed: {baseline_result.get('error', 'Unknown error')}")
    
    # Run local
    if not args.baseline_only:
        if not os.environ.get("LLM_GATEWAY_KEY"):
            print("\n--- Skipping Local (LLM_GATEWAY_KEY not set) ---")
        else:
            print("\n--- Running Local Generation (HIPGenerator) ---")
            local_result = run_local_generation(str(problem_path), args.max_retries)
            
            if local_result["success"]:
                status = "✓" if local_result["correct"] else "✗"
                print(f"  Result: {status} Compiled={local_result['compiled']}, "
                      f"Correct={local_result['correct']}, Speedup={local_result['speedup']:.2f}x")
            else:
                print(f"  Failed: {local_result.get('error', 'Unknown error')}")
    
    # Compare
    if baseline_result and local_result:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        
        baseline_ok = baseline_result.get("correct", False)
        local_ok = local_result.get("correct", False)
        
        baseline_speed = baseline_result.get("speedup", 0) if baseline_ok else 0
        local_speed = local_result.get("speedup", 0) if local_ok else 0
        
        print(f"  Baseline: {'✓' if baseline_ok else '✗'} ({baseline_speed:.2f}x)")
        print(f"  Local:    {'✓' if local_ok else '✗'} ({local_speed:.2f}x)")
        
        if baseline_speed > local_speed:
            print(f"  Winner: BASELINE (+{baseline_speed - local_speed:.2f}x)")
        elif local_speed > baseline_speed:
            print(f"  Winner: LOCAL (+{local_speed - baseline_speed:.2f}x)")
        else:
            print(f"  Winner: TIE")


if __name__ == "__main__":
    main()
