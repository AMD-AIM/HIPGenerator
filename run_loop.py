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
os.environ["TRITON_HIP_USE_BLOCK_PINGPONG"] = "1"  # Enable pingpong scheduling for Triton

# Supported backends
BACKENDS = ["hip", "triton"]


def repair_triton_code(code: str) -> str:
    """Repair Triton code by injecting missing optimizations.
    
    This function checks for critical MI350 optimizations and injects them if missing:
    1. XCD swizzle (critical for 32 XCDs on MI350)
    2. Environment variables for pingpong/async copy
    3. matrix_instr_nonkdim=16 for MFMA
    """
    import re
    
    repaired = code
    changes_made = []
    
    # Check and add environment variables at the top after imports
    if "TRITON_HIP_USE_BLOCK_PINGPONG" not in repaired:
        # Find first import line and add after it
        import_match = re.search(r'^import\s+\w+', repaired, re.MULTILINE)
        if import_match:
            insert_pos = repaired.find('\n', import_match.end()) + 1
            env_code = "\nimport os\nos.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'\nos.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'\n"
            repaired = repaired[:insert_pos] + env_code + repaired[insert_pos:]
            changes_made.append("Added TRITON_HIP_USE_BLOCK_PINGPONG and ASYNC_COPY")
    
    # Check and add XCD swizzle if missing
    if "NUM_XCDS" not in repaired and "pids_per_xcd" not in repaired:
        # Add NUM_XCDS constant if not present
        if "NUM_XCDS" not in repaired:
            # Find a good place to insert (after os.environ lines or before @triton.jit)
            jit_match = re.search(r'^@triton\.jit', repaired, re.MULTILINE)
            if jit_match:
                repaired = repaired[:jit_match.start()] + "NUM_XCDS = 32\n\n" + repaired[jit_match.start():]
                changes_made.append("Added NUM_XCDS = 32")
        
        # Now inject XCD swizzle into kernel after pid calculation
        # Look for the pattern: pid = tl.program_id(0)
        pid_pattern = r'(pid\s*=\s*tl\.program_id\(0\))'
        
        xcd_swizzle = '''\\1
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pids = num_pid_m * num_pid_n
    
    # XCD Swizzle for MI350's 32 chiplets
    pids_per_xcd = (num_pids + NUM_XCDS - 1) // NUM_XCDS
    xcd_id = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if local_pid < pids_per_xcd:
        remapped_pid = xcd_id * pids_per_xcd + local_pid
        if remapped_pid < num_pids:
            pid = remapped_pid'''
        
        if re.search(pid_pattern, repaired):
            # Check if num_pid_m already defined after pid
            if "num_pid_m" in repaired:
                # XCD swizzle code needs to be inserted between pid and num_pid_m
                # More complex insertion - find where to put XCD logic
                pass  # Skip if already has grouping logic
            else:
                repaired = re.sub(pid_pattern, xcd_swizzle, repaired, count=1)
                changes_made.append("Injected XCD swizzle")
    
    # Check and add matrix_instr_nonkdim=16 if missing
    if "matrix_instr_nonkdim" not in repaired:
        # Find kernel launch pattern and add the parameter
        launch_patterns = [
            r'(kernel\[.*?\]\(.*?num_warps\s*=\s*\d+)',
            r'(matmul_kernel\[.*?\]\(.*?num_warps\s*=\s*\d+)',
            r'(\w+_kernel\[.*?\]\(.*?num_warps\s*=\s*\d+)',
        ]
        for pattern in launch_patterns:
            if re.search(pattern, repaired, re.DOTALL):
                repaired = re.sub(
                    pattern + r'(\s*,?\s*\))',
                    r'\1, matrix_instr_nonkdim=16\2',
                    repaired,
                    count=1,
                    flags=re.DOTALL
                )
                changes_made.append("Added matrix_instr_nonkdim=16")
                break
    
    if changes_made:
        print(f"  Code repaired: {', '.join(changes_made)}")
    
    return repaired


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
            base_prompt_path = script_dir / "prompts" / "triton_gemm_mi350_golden.txt"
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
    
    # Repair Triton code if needed (inject XCD swizzle, etc.)
    if backend == "triton":
        for sample_path in samples:
            try:
                with open(sample_path, 'r') as f:
                    code = f.read()
                repaired_code = repair_triton_code(code)
                if repaired_code != code:
                    with open(sample_path, 'w') as f:
                        f.write(repaired_code)
            except Exception as e:
                print(f"  Warning: Could not repair {sample_path}: {e}")
    
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
    """Build feedback string from evaluation results with detailed profiler metrics.
    
    This feedback is structured to help the LLM understand exactly what went wrong
    and what specific code changes are needed based on proven golden solutions.
    """
    feedback_parts = []
    
    best_speedup = max((r.get("speedup", 0) for r in results if r and r.get("accuracy_pass")), default=0)
    best_result = None
    for r in results:
        if r and r.get("accuracy_pass") and r.get("speedup", 0) == best_speedup:
            best_result = r
            break
    
    # Categorize all results
    compile_errors = []
    accuracy_failures = []
    performance_issues = []
    
    for i, res in enumerate(results):
        if not res:
            continue
        if not res.get("compile_success"):
            compile_errors.append((i+1, res))
        elif not res.get("accuracy_pass"):
            accuracy_failures.append((i+1, res))
        else:
            performance_issues.append((i+1, res))
    
    # Handle compile errors with specific fixes
    if compile_errors:
        feedback_parts.append("## COMPILE ERRORS (MUST FIX FIRST)")
        for idx, res in compile_errors:
            error = res.get("error", "Compilation failed")
            # Extract most relevant error lines
            if "error:" in error.lower():
                lines = [l.strip() for l in error.split('\n') if 'error:' in l.lower()]
                error_excerpt = '\n'.join(lines[:3])
            else:
                error_excerpt = error[:400]
            
            feedback_parts.append(f"\nSample {idx}: {error_excerpt}")
            
            if backend == "triton":
                # Provide specific fixes based on error patterns
                if "unexpected keyword argument 'matrix_instr_nonkdim'" in error:
                    feedback_parts.append("FIX: matrix_instr_nonkdim goes in kernel LAUNCH, not triton.Config:")
                    feedback_parts.append("  kernel[grid](..., num_stages=2, num_warps=8, matrix_instr_nonkdim=16)")
                if "constexpr" in error.lower():
                    feedback_parts.append("FIX: Ensure all tl.constexpr params are defined as such in signature")
                if "stride" in error.lower():
                    feedback_parts.append("FIX: Check stride parameter count matches kernel signature")
                if "cannot find" in error.lower() or "undefined" in error.lower():
                    feedback_parts.append("FIX: Check all imports - need 'import triton' and 'import triton.language as tl'")
    
    # Handle accuracy failures with specific debugging guidance
    if accuracy_failures:
        feedback_parts.append("\n## ACCURACY FAILURES (WRONG OUTPUT)")
        for idx, res in accuracy_failures:
            max_diff = res.get("max_diff", "N/A")
            has_nan = res.get("has_nan", False)
            has_inf = res.get("has_inf", False)
            
            if has_nan:
                feedback_parts.append(f"\nSample {idx}: OUTPUT HAS NaN VALUES!")
                feedback_parts.append("DIAGNOSIS: NaN usually means reading uninitialized memory or division by zero")
                feedback_parts.append("FIXES:")
                feedback_parts.append("  1. Check offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)")
                feedback_parts.append("  2. Check offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)")
                feedback_parts.append("  3. Check k_mask = (k + tl.arange(0, BLOCK_K)) < K  # NOT offs_k!")
                feedback_parts.append("  4. Check pointer setup: a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak")
                feedback_parts.append("  5. For nn.Linear weight transpose: swap stride_wk and stride_wn")
            elif has_inf:
                feedback_parts.append(f"\nSample {idx}: OUTPUT HAS Inf VALUES!")
                feedback_parts.append("DIAGNOSIS: Inf means numerical overflow")
                feedback_parts.append("FIXES:")
                feedback_parts.append("  1. Use float32 accumulator: acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)")
                feedback_parts.append("  2. Convert only at store: tl.store(c_ptrs, acc.to(tl.float16), mask=mask)")
            else:
                feedback_parts.append(f"\nSample {idx}: max_diff = {max_diff} (too high!)")
                feedback_parts.append("DIAGNOSIS: Likely wrong matrix layout or stride handling")
                feedback_parts.append("FIXES for GEMM (A @ B):")
                feedback_parts.append("  - stride_am = A.stride(0), stride_ak = A.stride(1)")
                feedback_parts.append("  - stride_bk = B.stride(0), stride_bn = B.stride(1)")
                feedback_parts.append("FIXES for nn.Linear (x @ W.T):")
                feedback_parts.append("  - Weight is [out_features, in_features] = [N, K]")
                feedback_parts.append("  - To treat as [K, N] transposed: stride_wk = 1, stride_wn = K")
                feedback_parts.append("  - Then: w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk")
                feedback_parts.append("FIXES for A.T @ B:")
                feedback_parts.append("  - A is [K, M], swap strides: stride_am = A.stride(1), stride_ak = A.stride(0)")
    
    # Handle performance issues with proven optimization techniques
    if performance_issues:
        feedback_parts.append("\n## PERFORMANCE ANALYSIS")
        
        for idx, res in performance_issues:
            speedup = res.get("speedup", 0)
            ref_time = res.get("ref_time_ms", 0)
            new_time = res.get("new_time_ms", 0)
            rocprof = res.get("rocprof_metrics", {})
            
            feedback_parts.append(f"\nSample {idx}: {speedup:.3f}x speedup (ref={ref_time:.3f}ms, yours={new_time:.3f}ms)")
            
            if backend == "triton":
                if speedup < 0.5:
                    feedback_parts.append("\n**CRITICAL: Kernel is 2x+ slower than baseline!**")
                    feedback_parts.append("YOU MUST ADD THESE OPTIMIZATIONS:")
                    feedback_parts.append("")
                    feedback_parts.append("1. XCD SWIZZLE (MI350 has 32 XCDs - without this you get ~0.3x!):")
                    feedback_parts.append("   NUM_XCDS = 32")
                    feedback_parts.append("   pids_per_xcd = (num_pids + NUM_XCDS - 1) // NUM_XCDS")
                    feedback_parts.append("   xcd_id = pid % NUM_XCDS")
                    feedback_parts.append("   local_pid = pid // NUM_XCDS")
                    feedback_parts.append("   if local_pid < pids_per_xcd:")
                    feedback_parts.append("       remapped_pid = xcd_id * pids_per_xcd + local_pid")
                    feedback_parts.append("       if remapped_pid < num_pids:")
                    feedback_parts.append("           pid = remapped_pid")
                    feedback_parts.append("")
                    feedback_parts.append("2. Use 16x16 MFMA instructions:")
                    feedback_parts.append("   kernel[grid](..., matrix_instr_nonkdim=16)")
                    feedback_parts.append("")
                    feedback_parts.append("3. Use L2 grouping:")
                    feedback_parts.append("   num_pid_in_group = GROUP_M * num_pid_n")
                    feedback_parts.append("   group_id = pid // num_pid_in_group")
                    feedback_parts.append("   first_pid_m = group_id * GROUP_M")
                    feedback_parts.append("   group_size_m = min(num_pid_m - first_pid_m, GROUP_M)")
                    feedback_parts.append("   pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)")
                    feedback_parts.append("   pid_n = (pid % num_pid_in_group) // group_size_m")
                    
                elif speedup < 0.8:
                    feedback_parts.append("\n**MODERATE: Need additional optimizations**")
                    feedback_parts.append("ADD THESE:")
                    feedback_parts.append("1. Enable pingpong at TOP of file:")
                    feedback_parts.append("   os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'")
                    feedback_parts.append("   os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'")
                    feedback_parts.append("")
                    feedback_parts.append("2. Use optimal block sizes:")
                    feedback_parts.append("   - For square M,N>=4096: BLOCK_M=256, BLOCK_N=256, BLOCK_K=32, stages=3, warps=8")
                    feedback_parts.append("   - For large K>max(M,N): BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, stages=2, warps=8")
                    feedback_parts.append("   - Set GROUP_M=8 or 16")
                    
                elif speedup < 1.0:
                    feedback_parts.append("\n**CLOSE: Need launch overhead elimination**")
                    feedback_parts.append("MOVE THESE TO __init__:")
                    feedback_parts.append("1. Precompute grid:")
                    feedback_parts.append("   self._grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)")
                    feedback_parts.append("2. Preallocate output:")
                    feedback_parts.append("   self.register_buffer('_out', torch.empty((M, N), dtype=torch.float16))")
                    feedback_parts.append("3. Precompute ALL strides (no .stride() in forward!):")
                    feedback_parts.append("   self._stride_am = K; self._stride_ak = 1; ...")
                    
                elif speedup < 1.1:
                    feedback_parts.append("\n**ALMOST THERE: Fine-tuning needed**")
                    feedback_parts.append("TRY:")
                    feedback_parts.append("1. Increase num_stages to 3 (from 2)")
                    feedback_parts.append("2. Try GROUP_M=16 instead of 8")
                    feedback_parts.append("3. For aligned dimensions, remove all masks for max speed")
                else:
                    feedback_parts.append(f"\n**GOOD: {speedup:.2f}x exceeds target!**")
    
    # Overall summary
    if best_speedup > 0:
        feedback_parts.append(f"\n## SUMMARY")
        feedback_parts.append(f"Best speedup achieved: {best_speedup:.3f}x (target: >= 1.1x)")
        
        if best_speedup < 1.1:
            feedback_parts.append("\nPRIORITY ACTIONS:")
            if best_speedup < 0.5:
                feedback_parts.append("  1. ADD XCD swizzle code (mandatory for MI350)")
                feedback_parts.append("  2. ADD matrix_instr_nonkdim=16 to kernel launch")
                feedback_parts.append("  3. Use tl.dot() for matrix multiply")
            elif best_speedup < 0.8:
                feedback_parts.append("  1. Set TRITON_HIP_USE_BLOCK_PINGPONG='1' at file top")
                feedback_parts.append("  2. Use optimal block sizes from table")
            elif best_speedup < 1.0:
                feedback_parts.append("  1. Move grid/stride computation to __init__")
                feedback_parts.append("  2. Use register_buffer for output")
            else:
                feedback_parts.append("  1. Try num_stages=3, GROUP_M=16")
                feedback_parts.append("  2. Fuse any post-GEMM ops")
    elif not compile_errors and not accuracy_failures:
        feedback_parts.append("\nNo successful samples - check compile/accuracy errors above")
    
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
    parser.add_argument("--target-speedup", type=float, default=1.1, help="Target speedup (default 1.1x, achievable with optimized kernel)")
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

