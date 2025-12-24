#!/usr/bin/env python3
"""
Generate-Evaluate-Profile Loop for HipKittens/Triton Kernel Generation.

This script implements a complete feedback loop:
1. Generate code using LLM
2. Evaluate code (compile, accuracy, performance)
3. Profile with rocprof to get optimization hints
4. Feed hints back to LLM for re-generation if needed

Usage:
    python run_loop.py --problem <problem_path> --max-attempts 3 [--backend hip|triton]
"""
import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

# Supported backends
BACKENDS = ["hip", "triton"]


def run_generate(problem_path: str, output_path: str, num_samples: int = 3, 
                 feedback: str = None, attempt: int = 1, backend: str = "hip") -> list:
    """Run generate.py and return paths to generated samples."""
    script_dir = Path(__file__).parent
    
    # Build user feedback if provided
    extra_prompt = ""
    if feedback:
        extra_prompt = f"\n\n**PREVIOUS ATTEMPT FEEDBACK (MUST ADDRESS!):**\n{feedback}\n"
    
    # Create temp prompt file with feedback if needed
    prompt_file = None
    if extra_prompt:
        import tempfile
        # Load base prompt based on backend
        if backend == "triton":
            base_prompt_path = script_dir / "prompts" / "triton_gemm.txt"
        else:
            base_prompt_path = script_dir / "prompts" / "hipkittens_gemm_v4.txt"
        
        if base_prompt_path.exists():
            base_prompt = base_prompt_path.read_text()
        else:
            base_prompt = ""
        
        # Append feedback
        combined_prompt = base_prompt + extra_prompt
        
        # Write to temp file
        fd, prompt_file = tempfile.mkstemp(suffix='.txt')
        os.write(fd, combined_prompt.encode())
        os.close(fd)
    
    cmd = [
        sys.executable, str(script_dir / "generate.py"),
        "--problem", problem_path,
        "--output", output_path,
        "--num-samples", str(num_samples),
        "--temperature", "0.3",  # Higher temp for diversity
        "--backend", backend,
    ]
    if prompt_file:
        cmd.extend(["--prompt", prompt_file])
    
    print(f"\n{'='*60}")
    print(f"GENERATE (Attempt {attempt})")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False, env=os.environ)
    
    # Cleanup temp file
    if prompt_file and os.path.exists(prompt_file):
        os.unlink(prompt_file)
    
    if result.returncode != 0:
        print(f"Generation failed with return code {result.returncode}")
        return []
    
    # Find generated sample files
    output_dir = Path(output_path).parent
    base_name = Path(output_path).stem
    samples = []
    
    # First check for the main file (samples=1 case)
    main_file = Path(output_path)
    if main_file.exists() and main_file.stat().st_size > 0:
        samples.append(str(main_file))
    
    # Then check for _s{i} files (samples>1 case)
    for i in range(1, num_samples + 1):
        sample_path = output_dir / f"{base_name}_s{i}.py"
        if sample_path.exists() and str(sample_path) not in samples:
            samples.append(str(sample_path))
    
    print(f"Generated {len(samples)} samples")
    return samples


def run_evaluate(code_path: str, problem_path: str, output_path: str, backend: str = "hip", 
                 run_profile: bool = True) -> dict:
    """Run eval.py and return results with profile info."""
    script_dir = Path(__file__).parent
    
    cmd = [
        sys.executable, str(script_dir / "eval.py"),
        "--code", code_path,
        "--problem", problem_path,
        "--output", output_path,
        "--backend", backend,
    ]
    
    # Enable profiling for performance analysis
    if run_profile:
        cmd.append("--profile")
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    
    # Load results
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    
    return {
        "compile_success": False,
        "accuracy_pass": False,
        "error": result.stderr or "Unknown error",
        "speedup": 0.0,
    }


def build_feedback(results: list, backend: str = "hip") -> str:
    """Build feedback string from evaluation results with detailed profiler metrics."""
    feedback_parts = []
    
    best_speedup = max((r.get("speedup", 0) for r in results if r and r.get("accuracy_pass")), default=0)
    best_result = None
    for r in results:
        if r and r.get("accuracy_pass") and r.get("speedup", 0) == best_speedup:
            best_result = r
            break
    
    for i, res in enumerate(results):
        if not res:
            continue
            
        # Compile errors
        if not res.get("compile_success"):
            error = res.get("error", "Compilation failed")
            # Extract key error message
            if "error:" in error.lower():
                lines = [l for l in error.split('\n') if 'error:' in l.lower()]
                error = '\n'.join(lines[:3])  # First 3 error lines
            feedback_parts.append(f"Sample {i+1} compile error: {error[:500]}")
            continue
        
        # Accuracy failures
        if not res.get("accuracy_pass"):
            max_diff = res.get("max_diff", "N/A")
            if res.get("has_nan"):
                feedback_parts.append(f"Sample {i+1}: OUTPUT HAS NaN! Check memory access and kernel logic.")
            elif res.get("has_inf"):
                feedback_parts.append(f"Sample {i+1}: OUTPUT HAS Inf! Check for overflow.")
            else:
                feedback_parts.append(f"Sample {i+1}: Accuracy failed (max_diff={max_diff}). Check GEMM logic.")
            continue
        
        # Performance issues with detailed profiler metrics
        speedup = res.get("speedup", 0)
        ref_time = res.get("ref_time_ms", 0)
        new_time = res.get("new_time_ms", 0)
        rocprof = res.get("rocprof_metrics", {})
        
        feedback_parts.append(f"**Sample {i+1} Performance:**")
        feedback_parts.append(f"  - Speedup: {speedup:.3f}x (target: > 0.8x)")
        feedback_parts.append(f"  - Reference time: {ref_time:.3f}ms, Your kernel: {new_time:.3f}ms")
        
        # Include profiler metrics if available
        kernel_name = rocprof.get("kernel_name")
        if kernel_name:
            feedback_parts.append(f"  - Kernel: {kernel_name[:60]}")
        
        avg_duration = rocprof.get("avg_duration_us", 0)
        if avg_duration > 0:
            feedback_parts.append(f"  - Kernel avg time: {avg_duration:.1f}us")
        
        lds_bytes = rocprof.get("lds_usage_bytes", 0)
        if lds_bytes > 0:
            feedback_parts.append(f"  - LDS usage: {lds_bytes/1024:.1f}KB")
        
        wg_size = rocprof.get("workgroup_size", [])
        if wg_size and wg_size[0] > 0:
            threads = wg_size[0] * wg_size[1] * wg_size[2]
            feedback_parts.append(f"  - Workgroup: {wg_size[0]}x{wg_size[1]}x{wg_size[2]} = {threads} threads")
        
        grid_size = rocprof.get("grid_size", [])
        if grid_size and grid_size[0] > 0:
            feedback_parts.append(f"  - Grid: {grid_size[0]}x{grid_size[1]}x{grid_size[2]}")
        
        # Optimization hints based on profiler data
        if backend == "triton":
            feedback_parts.append("\n**Optimization Suggestions:**")
            
            if speedup < 0.3:
                feedback_parts.append("  CRITICAL ISSUE - kernel is 3x+ slower than baseline:")
                feedback_parts.append("    1. Check if using tl.dot() for matrix multiply (NOT loops)")
                feedback_parts.append("    2. Ensure BLOCK_K >= 64, ideally 128 for AMD MFMA")
                feedback_parts.append("    3. Remove unnecessary memory operations")
            elif speedup < 0.6:
                feedback_parts.append("  MODERATE ISSUE - try these optimizations:")
                feedback_parts.append("    1. Increase tile sizes: BLOCK_M=256, BLOCK_N=256")
                feedback_parts.append("    2. Use num_warps=8 for large tiles")
                feedback_parts.append("    3. Reduce num_stages to 2 for less register pressure")
            elif speedup < 0.8:
                feedback_parts.append("  MINOR TUNING needed:")
                feedback_parts.append("    1. Fine-tune GROUP_SIZE_M (try 4 or 8)")
                feedback_parts.append("    2. Try BLOCK_K=128 if not already")
                feedback_parts.append("    3. Ensure proper memory coalescing")
            
            # Specific hints from profiler
            hints = rocprof.get("optimization_hints", [])
            if hints:
                feedback_parts.append(f"\n  Profiler hints: {'; '.join(hints)}")
    
    # Overall summary
    if best_speedup > 0:
        feedback_parts.append(f"\n**SUMMARY: Best speedup = {best_speedup:.3f}x**")
        if best_speedup < 0.5:
            feedback_parts.append("Focus on fundamental algorithm changes (tile sizes, BLOCK_K).")
        elif best_speedup < 0.8:
            feedback_parts.append("Getting close! Focus on autotune config optimization.")
    
    if not feedback_parts:
        return None
    
    return "\n".join(feedback_parts)


def run_loop(problem_path: str, output_dir: str, max_attempts: int = 3, 
             samples_per_attempt: int = 3, target_speedup: float = 0.9,
             backend: str = "hip") -> dict:
    """Run the complete generate-evaluate-profile loop."""
    
    problem_name = Path(problem_path).stem
    problem_dir = Path(output_dir) / problem_name
    problem_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print(f"# PROBLEM: {problem_name}")
    print(f"# Backend: {backend}")
    print(f"# Target speedup: >= {target_speedup}x")
    print(f"# Max attempts: {max_attempts}")
    print(f"{'#'*70}")
    
    best_result = None
    best_speedup = 0.0
    best_code_path = None
    feedback = None
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n{'='*60}")
        print(f"ATTEMPT {attempt}/{max_attempts}")
        print(f"{'='*60}")
        
        # Generate samples
        output_path = str(problem_dir / f"code_{attempt}.py")
        samples = run_generate(
            problem_path, output_path, 
            num_samples=samples_per_attempt,
            feedback=feedback,
            attempt=attempt,
            backend=backend
        )
        
        if not samples:
            print(f"No samples generated in attempt {attempt}")
            continue
        
        # Evaluate each sample
        results = []
        for i, sample_path in enumerate(samples):
            print(f"\n--- Evaluating sample {i+1}: {Path(sample_path).name} ---")
            result_path = str(problem_dir / f"result_{attempt}_s{i}.json")
            
            result = run_evaluate(sample_path, problem_path, result_path, backend=backend)
            result["sample_path"] = sample_path
            results.append(result)
            
            # Print summary
            compile_ok = "✓" if result.get("compile_success") else "✗"
            accuracy_ok = "✓" if result.get("accuracy_pass") else "✗"
            speedup = result.get("speedup", 0)
            
            print(f"  Compile: {compile_ok} | Accuracy: {accuracy_ok} | Speedup: {speedup:.2f}x")
            
            if result.get("rocprof_metrics", {}).get("optimization_hints"):
                for hint in result["rocprof_metrics"]["optimization_hints"]:
                    print(f"  Profile: {hint}")
            
            # Check if this is the best so far
            if result.get("accuracy_pass") and speedup > best_speedup:
                best_speedup = speedup
                best_result = result
                best_code_path = sample_path
                
                # Copy best code
                shutil.copy(sample_path, problem_dir / "best_code.py")
                with open(problem_dir / "best_result.json", "w") as f:
                    json.dump(result, f, indent=2)
        
        # Check if we've achieved target
        if best_speedup >= target_speedup:
            print(f"\n✓ Target speedup achieved: {best_speedup:.2f}x >= {target_speedup}x")
            break
        
        # Build feedback for next attempt
        feedback = build_feedback(results, backend=backend)
        if feedback:
            print(f"\n--- Feedback for next attempt ---\n{feedback}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL RESULT for {problem_name}")
    print(f"{'='*70}")
    
    if best_result:
        print(f"Best speedup: {best_speedup:.2f}x")
        print(f"Best code: {best_code_path}")
        print(f"Accuracy: {'PASS' if best_result.get('accuracy_pass') else 'FAIL'}")
        
        status = "success" if best_speedup >= target_speedup else "partial"
    else:
        print("No working solution found!")
        status = "failed"
    
    return {
        "problem": problem_name,
        "status": status,
        "best_speedup": best_speedup,
        "best_code": best_code_path,
        "best_result": best_result,
        "attempts": attempt,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate-Evaluate-Profile Loop")
    parser.add_argument("--problem", required=True, help="Problem file path or comma-separated list")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max attempts per problem")
    parser.add_argument("--samples", type=int, default=1, help="Samples per attempt (1 for focused iteration)")
    parser.add_argument("--target-speedup", type=float, default=0.5, help="Target speedup (default 0.5x, realistic for Triton on AMD)")
    parser.add_argument("--backend", choices=BACKENDS, default="hip",
                        help="Backend type: 'hip' for HipKittens, 'triton' for Triton (default: hip)")
    args = parser.parse_args()
    
    # Check LLM key
    if not os.environ.get("LLM_GATEWAY_KEY"):
        print("Error: LLM_GATEWAY_KEY not set")
        sys.exit(1)
    
    print(f"Using backend: {args.backend}")
    
    # Handle multiple problems
    if ',' in args.problem:
        problems = [p.strip() for p in args.problem.split(',')]
    else:
        problems = [args.problem]
    
    # Run for each problem
    all_results = []
    for problem in problems:
        if not os.path.exists(problem):
            print(f"Problem not found: {problem}")
            continue
        
        result = run_loop(
            problem, 
            args.output,
            max_attempts=args.max_attempts,
            samples_per_attempt=args.samples,
            target_speedup=args.target_speedup,
            backend=args.backend
        )
        all_results.append(result)
    
    # Summary
    print(f"\n{'#'*70}")
    print("OVERALL SUMMARY")
    print(f"{'#'*70}")
    
    success = sum(1 for r in all_results if r["status"] == "success")
    partial = sum(1 for r in all_results if r["status"] == "partial")
    failed = sum(1 for r in all_results if r["status"] == "failed")
    
    print(f"Total: {len(all_results)}")
    print(f"  ✓ Success (>= target): {success}")
    print(f"  ⚠ Partial (accuracy pass): {partial}")
    print(f"  ✗ Failed: {failed}")
    
    for r in all_results:
        status_sym = "✓" if r["status"] == "success" else ("⚠" if r["status"] == "partial" else "✗")
        print(f"  {status_sym} {r['problem']}: {r['best_speedup']:.2f}x ({r['attempts']} attempts)")
    
    # Save summary
    summary_path = Path(args.output) / "loop_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

