#!/usr/bin/env python3
"""
Generate-Evaluate-Profile Loop for HipKittens Kernel Generation.

This script implements a complete feedback loop:
1. Generate code using LLM
2. Evaluate code (compile, accuracy, performance)
3. Profile with rocprof to get optimization hints
4. Feed hints back to LLM for re-generation if needed

Usage:
    python run_loop.py --problem <problem_path> --max-attempts 3
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


def run_generate(problem_path: str, output_path: str, num_samples: int = 3, 
                 feedback: str = None, attempt: int = 1) -> list:
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
        # Load base prompt
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
        "--temperature", "0.1",  # Low temp for focused generation
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


def run_evaluate(code_path: str, problem_path: str, output_path: str) -> dict:
    """Run eval.py and return results with profile info."""
    script_dir = Path(__file__).parent
    
    cmd = [
        sys.executable, str(script_dir / "eval.py"),
        "--code", code_path,
        "--problem", problem_path,
        "--output", output_path,
    ]
    
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


def build_feedback(results: list) -> str:
    """Build feedback string from evaluation results."""
    feedback_parts = []
    
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
        
        # Performance issues
        speedup = res.get("speedup", 0)
        if speedup < 1.0:
            hints = res.get("rocprof_metrics", {}).get("optimization_hints", [])
            perf_analysis = res.get("perf_analysis", "")
            
            feedback_parts.append(f"Sample {i+1}: Accuracy PASS but speedup={speedup:.2f}x < 1.0x")
            if hints:
                feedback_parts.append(f"  Profiler hints: {'; '.join(hints)}")
            if "HIGH LDS" in perf_analysis:
                feedback_parts.append("  Consider reducing tile size or shared memory usage")
    
    if not feedback_parts:
        return None
    
    return "\n".join(feedback_parts)


def run_loop(problem_path: str, output_dir: str, max_attempts: int = 3, 
             samples_per_attempt: int = 3, target_speedup: float = 0.9) -> dict:
    """Run the complete generate-evaluate-profile loop."""
    
    problem_name = Path(problem_path).stem
    problem_dir = Path(output_dir) / problem_name
    problem_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print(f"# PROBLEM: {problem_name}")
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
            attempt=attempt
        )
        
        if not samples:
            print(f"No samples generated in attempt {attempt}")
            continue
        
        # Evaluate each sample
        results = []
        for i, sample_path in enumerate(samples):
            print(f"\n--- Evaluating sample {i+1}: {Path(sample_path).name} ---")
            result_path = str(problem_dir / f"result_{attempt}_s{i}.json")
            
            result = run_evaluate(sample_path, problem_path, result_path)
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
        feedback = build_feedback(results)
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
    parser.add_argument("--samples", type=int, default=3, help="Samples per attempt")
    parser.add_argument("--target-speedup", type=float, default=0.9, help="Target speedup (default 0.9x)")
    args = parser.parse_args()
    
    # Check LLM key
    if not os.environ.get("LLM_GATEWAY_KEY"):
        print("Error: LLM_GATEWAY_KEY not set")
        sys.exit(1)
    
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
            target_speedup=args.target_speedup
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

