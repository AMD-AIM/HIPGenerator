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
    1. Environment variables for pingpong/async copy
    2. XCD swizzle (ONLY for GEMM/matmul, NOT for element-wise ops)
    3. matrix_instr_nonkdim=16 for MFMA (ONLY for GEMM/matmul)
    """
    import re
    
    repaired = code
    changes_made = []
    
    # Detect if this is a GEMM/matmul kernel vs element-wise
    code_lower = code.lower()
    is_gemm = any(op in code_lower for op in [
        'tl.dot(', 'matmul', 'gemm', 'mm(', 'linear',
        'stride_am', 'stride_bk', 'block_m', 'block_n', 'block_k'
    ])
    is_elementwise = any(op in code_lower for op in [
        'relu', 'sigmoid', 'gelu', 'tanh', 'swish', 'silu', 'softmax',
        'exp(', 'log(', 'sqrt(', 'maximum(', 'minimum('
    ]) and not is_gemm
    
    # Check and add environment variables at the top after imports (for all kernels)
    if "TRITON_HIP_USE_BLOCK_PINGPONG" not in repaired:
        # Find first import line and add after it
        import_match = re.search(r'^import\s+\w+', repaired, re.MULTILINE)
        if import_match:
            insert_pos = repaired.find('\n', import_match.end()) + 1
            env_code = "\nimport os\nos.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'\nos.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'\n"
            repaired = repaired[:insert_pos] + env_code + repaired[insert_pos:]
            changes_made.append("Added TRITON_HIP_USE_BLOCK_PINGPONG and ASYNC_COPY")
    
    # XCD swizzle is ONLY for GEMM/matmul kernels, NOT for element-wise
    if is_gemm and not is_elementwise:
        # Check and add XCD swizzle if missing
        if "NUM_XCDS" not in repaired and "pids_per_xcd" not in repaired:
            # Add NUM_XCDS constant before @triton.jit (NOT between decorators!)
            if "NUM_XCDS" not in repaired:
                # Find position BEFORE @triton.autotune or @triton.jit
                autotune_match = re.search(r'^@triton\.autotune', repaired, re.MULTILINE)
                jit_match = re.search(r'^@triton\.jit', repaired, re.MULTILINE)
                
                if autotune_match:
                    insert_pos = autotune_match.start()
                elif jit_match:
                    insert_pos = jit_match.start()
                else:
                    insert_pos = None
                
                if insert_pos is not None:
                    repaired = repaired[:insert_pos] + "NUM_XCDS = 32\n\n" + repaired[insert_pos:]
                    changes_made.append("Added NUM_XCDS = 32")
            
            # Only inject XCD swizzle if BLOCK_M and BLOCK_N are present (GEMM kernel)
            if "BLOCK_M" in repaired and "BLOCK_N" in repaired:
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
                    if "num_pid_m" not in repaired:
                        repaired = re.sub(pid_pattern, xcd_swizzle, repaired, count=1)
                        changes_made.append("Injected XCD swizzle")
        
        # Check and add matrix_instr_nonkdim=16 if missing (GEMM only)
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


def run_generate(problem_path: str, output_path: str,
                 feedback: str = None, attempt: int = 1, backend: str = "hip") -> str:
    """Run generate.py and return path to generated code.
    
    Returns:
        Path to generated code file, or None if generation failed.
    """
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
        "--num-samples", "1",  # Always generate 1 sample per attempt
        "--temperature", "0.1",  # Lower temp for more deterministic output
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
        return None
    
    # Check if output file exists
    if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
        print("No code generated")
        return None
    
    # Repair Triton code if needed (inject XCD swizzle, etc.)
    if backend == "triton":
        try:
            with open(output_path, 'r') as f:
                code = f.read()
            repaired_code = repair_triton_code(code)
            if repaired_code != code:
                with open(output_path, 'w') as f:
                    f.write(repaired_code)
        except Exception as e:
            print(f"  Warning: Could not repair code: {e}")
    
    print(f"Generated: {output_path}")
    return output_path


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


def build_metrics_feedback(rocprof_metrics: dict, speedup: float, problem_type: str) -> list:
    """Build optimization feedback based on rocprof metrics."""
    hints = []
    
    if not rocprof_metrics:
        return hints
    
    # Extract metrics
    vgpr = rocprof_metrics.get("vgpr_count", 0)
    sgpr = rocprof_metrics.get("sgpr_count", 0)
    lds_bytes = rocprof_metrics.get("lds_usage_bytes", 0)
    workgroup = rocprof_metrics.get("workgroup_size", [])
    grid = rocprof_metrics.get("grid_size", [])
    
    # PMC counters
    valu_insts = rocprof_metrics.get("SQ_INSTS_VALU", 0)
    salu_insts = rocprof_metrics.get("SQ_INSTS_SALU", 0)
    lds_conflicts = rocprof_metrics.get("SQ_LDS_BANK_CONFLICT", 0)
    l2_hit = rocprof_metrics.get("TCC_HIT", 0)
    l2_miss = rocprof_metrics.get("TCC_MISS", 0)
    
    # Calculate derived metrics
    total_threads = workgroup[0] * workgroup[1] * workgroup[2] if len(workgroup) >= 3 else 0
    total_blocks = grid[0] * grid[1] * grid[2] if len(grid) >= 3 else (grid[0] if grid else 0)
    l2_hit_rate = (l2_hit / (l2_hit + l2_miss) * 100) if (l2_hit + l2_miss) > 0 else -1
    
    # --- VGPR/SGPR Analysis ---
    if vgpr > 128:
        hints.append(f"⚠ HIGH VGPR ({vgpr}): Limits occupancy. Reduce register pressure by:")
        hints.append("   - Split complex computations into smaller kernels")
        hints.append("   - Use fewer intermediate variables")
        hints.append("   - Consider smaller BLOCK_SIZE")
    elif vgpr > 0 and vgpr < 32:
        hints.append(f"✓ LOW VGPR ({vgpr}): Good register usage, room for more computation")
    
    # --- Workgroup Size Analysis ---
    if total_threads > 0 and total_threads < 256:
        hints.append(f"⚠ SMALL WORKGROUP ({total_threads} threads): Poor occupancy")
        hints.append("   - Increase BLOCK_SIZE to at least 256")
        hints.append("   - Use num_warps=4 or higher")
    
    # --- Grid Size Analysis (MI350 has 256 CUs across 32 XCDs) ---
    if total_blocks > 0 and total_blocks < 256:
        hints.append(f"⚠ LOW PARALLELISM ({total_blocks} blocks < 256 CUs)")
        hints.append("   - Consider processing more elements per kernel launch")
        hints.append("   - Or use persistent kernel pattern")
    
    # --- L2 Cache Analysis ---
    if l2_hit_rate >= 0:
        if l2_hit_rate < 30:
            hints.append(f"⚠ VERY LOW L2 HIT RATE ({l2_hit_rate:.1f}%): Memory bandwidth bottleneck")
            hints.append("   - For GEMM: Add L2 tile grouping (GROUP_M=8)")
            hints.append("   - Reorder memory accesses for locality")
            hints.append("   - Consider loop tiling strategies")
        elif l2_hit_rate < 60:
            hints.append(f"⚠ LOW L2 HIT RATE ({l2_hit_rate:.1f}%): Consider cache optimization")
            hints.append("   - Group tiles that share data")
    
    # --- LDS Analysis ---
    if lds_bytes > 0:
        lds_kb = lds_bytes / 1024
        if lds_kb > 48:
            hints.append(f"⚠ HIGH LDS USAGE ({lds_kb:.1f} KB): May limit occupancy")
            hints.append("   - MI350 has 64KB LDS per CU, >48KB limits to 1 workgroup")
    
    if lds_conflicts > 0:
        hints.append(f"⚠ LDS BANK CONFLICTS ({lds_conflicts}): Stalled cycles")
        hints.append("   - Pad shared memory arrays to avoid conflicts")
        hints.append("   - Use strided access patterns")
    
    # --- Instruction Mix Analysis ---
    if valu_insts > 0 and salu_insts > 0:
        valu_ratio = valu_insts / (valu_insts + salu_insts) * 100
        if valu_ratio < 70:
            hints.append(f"⚠ LOW VALU RATIO ({valu_ratio:.0f}%): Too much scalar overhead")
            hints.append("   - Vectorize more operations")
            hints.append("   - Move loop invariants outside kernel")
    
    return hints


def build_feedback(results: list, backend: str = "hip", problem_type: str = "unknown") -> str:
    """Build feedback string from evaluation results with detailed profiler metrics.
    
    Uses rocprof_metrics for data-driven optimization suggestions rather than
    hardcoded speedup thresholds.
    
    Args:
        results: List of evaluation results
        backend: Backend type (hip or triton)
        problem_type: Type of problem (gemm, elementwise, softmax, norm, pooling, reduction)
    """
    feedback_parts = []
    
    # Detect problem category
    is_gemm = problem_type in ["gemm", "matmul", "linear", "gemm_2d", "batched_gemm", "matvec"]
    is_elementwise = problem_type in ["activation", "elementwise"]
    is_softmax = problem_type == "softmax"
    is_norm = problem_type == "norm"
    is_reduction = problem_type in ["reduction", "pooling"]
    
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
    
    # Handle performance issues based on rocprof metrics
    if performance_issues:
        feedback_parts.append("\n## PERFORMANCE ANALYSIS (Based on rocprof metrics)")
        
        for idx, res in performance_issues:
            speedup = res.get("speedup", 0)
            ref_time = res.get("ref_time_ms", 0)
            new_time = res.get("new_time_ms", 0)
            rocprof = res.get("rocprof_metrics", {})
            
            feedback_parts.append(f"\nSample {idx}: {speedup:.3f}x speedup (ref={ref_time:.3f}ms, yours={new_time:.3f}ms)")
            
            # Get metrics-based feedback
            metrics_hints = build_metrics_feedback(rocprof, speedup, problem_type)
            
            if metrics_hints:
                feedback_parts.append("\n**Profiler-based Recommendations:**")
                feedback_parts.extend(metrics_hints)
            
            # Add problem-type specific tips if speedup is low
            if speedup < 1.0 and backend == "triton":
                feedback_parts.append(f"\n**{problem_type.upper()} Optimization Tips:**")
                
                if is_gemm:
                    feedback_parts.append("• Ensure XCD swizzle for MI350 (32 XCDs)")
                    feedback_parts.append("• Use tl.dot() with BLOCK_M=BLOCK_N=128/256")
                    feedback_parts.append("• Add L2 grouping (GROUP_M=8)")
                    feedback_parts.append("• Set matrix_instr_nonkdim=16 in kernel launch")
                    
                elif is_elementwise:
                    feedback_parts.append("• Use large BLOCK_SIZE (4096-8192) with @triton.autotune")
                    feedback_parts.append("• Avoid internal loops - vectorize entire block")
                    feedback_parts.append("• Fast GELU: x * tl.sigmoid(1.702 * x)")
                    feedback_parts.append("• Compute in fp32, store in fp16")
                    
                elif is_softmax:
                    feedback_parts.append("• Use fused single-load algorithm (BLOCK_SIZE >= row_length)")
                    feedback_parts.append("• Single pass: load → max → exp → sum → divide → store")
                    feedback_parts.append("• For large rows: use online softmax algorithm")
                    
                elif is_norm:
                    feedback_parts.append("• Process entire feature dimension in one block if possible")
                    feedback_parts.append("• Use online algorithm for variance computation")
                    feedback_parts.append("• Compute statistics in fp32, normalize in fp16")
                    feedback_parts.append("• Minimize reads by fusing mean/var computation")
                    
                elif is_reduction:
                    feedback_parts.append("• Use hierarchical reduction (warp → block → grid)")
                    feedback_parts.append("• tl.sum/tl.max are already optimized for reductions")
                    feedback_parts.append("• For large tensors: multiple kernel launches")
                    
            elif speedup >= 1.0:
                feedback_parts.append(f"✓ Good performance: {speedup:.2f}x")
    
    # Overall summary
    if best_speedup > 0:
        feedback_parts.append(f"\n## SUMMARY")
        feedback_parts.append(f"Best speedup: {best_speedup:.2f}x")
        
        if best_speedup >= 1.0:
            feedback_parts.append("✓ Target achieved! Focus on further optimization if needed.")
        else:
            feedback_parts.append(f"Target not met. Review profiler recommendations above.")
            
            # Include best result's rocprof summary if available
            if best_result and best_result.get("rocprof_metrics"):
                rm = best_result["rocprof_metrics"]
                summary_items = []
                if rm.get("vgpr_count"):
                    summary_items.append(f"VGPR={rm['vgpr_count']}")
                if rm.get("TCC_HIT") and rm.get("TCC_MISS"):
                    hr = rm["TCC_HIT"] / (rm["TCC_HIT"] + rm["TCC_MISS"]) * 100
                    summary_items.append(f"L2={hr:.0f}%")
                if rm.get("workgroup_size"):
                    ws = rm["workgroup_size"]
                    if len(ws) >= 3:
                        summary_items.append(f"WG={ws[0]*ws[1]*ws[2]}")
                if summary_items:
                    feedback_parts.append(f"Key metrics: {', '.join(summary_items)}")
    elif not compile_errors and not accuracy_failures:
        feedback_parts.append("\nNo successful samples - check compile/accuracy errors above")
    
    if not feedback_parts:
        return None
    
    return "\n".join(feedback_parts)


def run_loop(problem_path: str, output_dir: str, max_attempts: int = 3,
             target_speedup: float = 0.9, backend: str = "hip") -> dict:
    """Run the complete generate-evaluate-profile loop.
    
    Args:
        problem_path: Path to the problem file
        output_dir: Output directory for results
        max_attempts: Maximum number of generate-evaluate attempts
        target_speedup: Target speedup to achieve (default 0.9x)
        backend: Backend type ('hip' or 'triton')
        
    Returns:
        Dictionary with results
    """
    from generate import classify_problem
    
    problem_name = Path(problem_path).stem
    problem_dir = Path(output_dir) / problem_name
    problem_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and classify problem
    with open(problem_path, 'r') as f:
        problem_code = f.read()
    problem_type = classify_problem(problem_name, problem_code)
    
    print(f"\n{'#'*70}")
    print(f"# PROBLEM: {problem_name}")
    print(f"# Type: {problem_type}")
    print(f"# Backend: {backend}")
    print(f"# Target speedup: >= {target_speedup}x")
    print(f"# Max attempts: {max_attempts}")
    print(f"{'#'*70}")
    
    best_result = None
    best_speedup = 0.0
    best_code_path = None
    feedback = None
    all_results = []
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n{'='*60}")
        print(f"ATTEMPT {attempt}/{max_attempts}")
        print(f"{'='*60}")
        
        # Generate code (1 sample per attempt)
        output_path = str(problem_dir / f"code_{attempt}.py")
        code_path = run_generate(
            problem_path, output_path,
            feedback=feedback,
            attempt=attempt,
            backend=backend
        )
        
        if not code_path:
            print(f"No code generated in attempt {attempt}")
            continue
        
        # Evaluate
        print(f"\n--- Evaluating: {Path(code_path).name} ---")
        result_path = str(problem_dir / f"result_{attempt}.json")
        
        result = run_evaluate(code_path, problem_path, result_path, backend=backend)
        result["code_path"] = code_path
        all_results.append(result)
        
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
            best_code_path = code_path
            
            # Copy best code
            shutil.copy(code_path, problem_dir / "best_code.py")
            with open(problem_dir / "best_result.json", "w") as f:
                json.dump(result, f, indent=2)
        
        # Check if we've achieved target
        if best_speedup >= target_speedup:
            print(f"\n✓ Target speedup achieved: {best_speedup:.2f}x >= {target_speedup}x")
            break
        
        # Build feedback for next attempt
        feedback = build_feedback([result], backend=backend, problem_type=problem_type)
        if feedback:
            print(f"\n--- Feedback for next attempt ---\n{feedback[:500]}...")
    
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
        "problem_type": problem_type,
        "status": status,
        "best_speedup": best_speedup,
        "best_code": best_code_path,
        "best_result": best_result,
        "attempts": attempt,
        "all_results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate-Evaluate-Profile Loop")
    parser.add_argument("--problem", required=True, help="Problem file path or comma-separated list")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max attempts per problem")
    parser.add_argument("--target-speedup", type=float, default=1.0, help="Target speedup (default 1.0x)")
    parser.add_argument("--backend", choices=BACKENDS, default="triton",
                        help="Backend type: 'hip' for HipKittens, 'triton' for Triton (default: triton)")
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

