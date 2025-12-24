#!/usr/bin/env python3
"""
Evaluate generated HipKittens/Triton kernel code.
Usage: python eval.py --code <code_path> --problem <problem_path> [--output <result.json>] [--backend hip|triton]
"""
import os
import sys
import json
import time
import argparse
import traceback
import subprocess
import tempfile
from pathlib import Path

# Disable core dumps
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
import torch.nn as nn

# Supported backends
BACKENDS = ["hip", "triton"]


def load_problem_module(problem_path: str):
    """Load reference model from problem file."""
    with open(problem_path) as f:
        code = f.read()
    
    exec_globals = {'torch': torch, 'nn': nn}
    exec(code, exec_globals)
    
    return exec_globals


def load_generated_code(code_path: str, backend: str = "hip"):
    """Load and compile generated code.
    
    For HIP backend: compiles C++/HIP code via load_inline
    For Triton backend: imports as module (required for @triton.jit to get source)
    """
    if backend == "triton":
        # Triton's @triton.jit needs to read source code via inspect.getsourcelines()
        # This requires the code to be in an actual file that can be imported
        import importlib.util
        import uuid
        
        # Generate unique module name to avoid caching issues
        module_name = f"triton_gen_{uuid.uuid4().hex[:8]}"
        
        # Load the module from file
        spec = importlib.util.spec_from_file_location(module_name, code_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {code_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            # Clean up on failure
            if module_name in sys.modules:
                del sys.modules[module_name]
            raise
        
        # Return module's namespace as dict
        return {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}
    else:
        # HIP backend: use exec() as before
        with open(code_path) as f:
            code = f.read()
        
        exec_globals = {'torch': torch, 'nn': nn}
        exec(code, exec_globals)
        
        return exec_globals


def benchmark(fn, warmup=10, iterations=100):
    """Benchmark a function."""
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        _ = fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iterations * 1000  # ms


def run_rocprof_analysis(code_path: str, problem_path: str) -> dict:
    """Run rocprofv3 to get detailed kernel performance metrics for GEMM optimization."""
    perf_info = {
        "kernel_name": None,
        "duration_us": 0.0,
        "duration_ms": 0.0,
        "total_calls": 0,
        "avg_duration_us": 0.0,
        "percentage": 0.0,
        # Memory metrics
        "lds_usage_bytes": 0,
        "private_segment_size": 0,
        "group_segment_size": 0,
        # Grid/block config
        "grid_size": [],
        "workgroup_size": [],
        # Raw metrics for analysis
        "raw_metrics": {},
        "analysis": "",
        "optimization_hints": []
    }
    
    try:
        import sqlite3
        
        # Create profiling script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(f'''
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch
import torch.nn as nn

exec(open("{problem_path}").read())
exec(open("{code_path}").read())

torch.manual_seed(42)
init_inputs = get_init_inputs() if 'get_init_inputs' in dir() else []
model = ModelNew(*init_inputs).cuda() if init_inputs else ModelNew().cuda()

inputs = get_inputs()
inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

with torch.no_grad():
    for _ in range(10):
        _ = model(*inputs)
    torch.cuda.synchronize()
''')
            profile_script = f.name
        
        # Output database path
        db_path = tempfile.mktemp(suffix='_results.db')
        output_prefix = db_path.replace('_results.db', '')
        
        # Run rocprofv3 with kernel trace
        cmd = [
            'rocprofv3',
            '--hip-trace',
            '--kernel-trace', 
            '-o', output_prefix,
            '--',
            'python3', profile_script
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            env=os.environ
        )
        
        # Parse SQLite database
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Query kernel_summary for stats
            try:
                cursor.execute("SELECT * FROM kernel_summary")
                rows = cursor.fetchall()
                
                pytorch_internal = ['elementwise', 'at::native', 'c10::', 'vectorized', 'copy']
                
                for row in rows:
                    kernel_name = row[0] if row else ""
                    is_internal = any(pat in kernel_name for pat in pytorch_internal)
                    
                    if not is_internal and ('gemm' in kernel_name.lower() or 'mfma' in kernel_name.lower() or 'kernel' in kernel_name.lower()):
                        # kernel_summary columns: name, calls, DURATION(nsec), SQR(nsec), AVERAGE(nsec), PERCENT, MIN(nsec), MAX(nsec), VARIANCE, STD_DEV
                        # Note: All time units are nanoseconds!
                        perf_info["kernel_name"] = kernel_name[:100]
                        perf_info["total_calls"] = int(row[1]) if len(row) > 1 else 0
                        duration_ns = float(row[2]) if len(row) > 2 else 0
                        perf_info["duration_us"] = duration_ns / 1000.0  # Convert ns to us
                        perf_info["duration_ms"] = duration_ns / 1e6     # Convert ns to ms
                        avg_ns = float(row[4]) if len(row) > 4 else 0
                        perf_info["avg_duration_us"] = avg_ns / 1000.0   # Convert ns to us
                        perf_info["percentage"] = float(row[5]) if len(row) > 5 else 0
                        
                        min_ns = float(row[6]) if len(row) > 6 else 0
                        max_ns = float(row[7]) if len(row) > 7 else 0
                        perf_info["raw_metrics"]["min_us"] = min_ns / 1000.0
                        perf_info["raw_metrics"]["max_us"] = max_ns / 1000.0
                        perf_info["raw_metrics"]["stddev_us"] = float(row[9]) / 1000.0 if len(row) > 9 and row[9] else 0
                        break
            except Exception as e:
                perf_info["raw_metrics"]["query_error"] = str(e)
            
            # Query kernel dispatch for grid/block sizes and LDS usage
            try:
                cursor.execute("""
                    SELECT group_segment_size, private_segment_size, 
                           workgroup_size_x, workgroup_size_y, workgroup_size_z,
                           grid_size_x, grid_size_y, grid_size_z
                    FROM rocpd_kernel_dispatch 
                    LIMIT 10
                """)
                dispatch_rows = cursor.fetchall()
                
                # Find our kernel's dispatch (typically has non-zero group_segment_size for GEMM)
                for drow in dispatch_rows:
                    if drow[0] > 0:  # group_segment_size > 0 indicates our kernel
                        perf_info["group_segment_size"] = drow[0]
                        perf_info["private_segment_size"] = drow[1]
                        perf_info["workgroup_size"] = [drow[2], drow[3], drow[4]]
                        perf_info["grid_size"] = [drow[5], drow[6], drow[7]]
                        perf_info["lds_usage_bytes"] = drow[0]
                        break
                        
            except Exception as e:
                perf_info["raw_metrics"]["dispatch_error"] = str(e)
            
            conn.close()
        
        # Generate optimization hints
        hints = []
        
        # Check LDS usage
        lds_bytes = perf_info.get("lds_usage_bytes", 0)
        if lds_bytes > 0:
            lds_kb = lds_bytes / 1024
            if lds_kb > 48:  # Using more than 48KB might limit occupancy
                hints.append(f"HIGH LDS USAGE ({lds_kb:.1f}KB): May limit occupancy, consider smaller tiles")
            perf_info["raw_metrics"]["lds_kb"] = lds_kb
        
        # Check workgroup size
        wg_size = perf_info.get("workgroup_size", [0, 0, 0])
        total_threads = wg_size[0] * wg_size[1] * wg_size[2] if wg_size else 0
        if total_threads > 0 and total_threads < 256:
            hints.append(f"SMALL WORKGROUP ({total_threads} threads): Consider larger blocks for better occupancy")
        
        # Check kernel percentage
        if perf_info["percentage"] < 80:
            hints.append(f"KERNEL ONLY {perf_info['percentage']:.1f}% of time: Check for host-side overhead")
        
        # Check variance
        min_us = perf_info["raw_metrics"].get("min_us", 0)
        max_us = perf_info["raw_metrics"].get("max_us", 0)
        if min_us > 0 and max_us / min_us > 1.5:
            hints.append(f"HIGH VARIANCE (min={min_us:.1f}us, max={max_us:.1f}us): Possible load imbalance")
        
        perf_info["optimization_hints"] = hints
        
        # Build analysis summary
        analysis_parts = []
        if perf_info["kernel_name"]:
            # Truncate kernel name for display
            name_short = perf_info["kernel_name"].split('(')[0][:40]
            analysis_parts.append(f"Kernel: {name_short}")
        if perf_info["avg_duration_us"] > 0:
            analysis_parts.append(f"Avg: {perf_info['avg_duration_us']:.1f}us")
        if perf_info["percentage"] > 0:
            analysis_parts.append(f"GPU%: {perf_info['percentage']:.1f}%")
        if perf_info["lds_usage_bytes"] > 0:
            analysis_parts.append(f"LDS: {perf_info['lds_usage_bytes']/1024:.1f}KB")
        if perf_info["workgroup_size"]:
            wg = perf_info["workgroup_size"]
            analysis_parts.append(f"Block: {wg[0]}x{wg[1]}x{wg[2]}")
        if hints:
            analysis_parts.append(f"Hints: {'; '.join(hints[:2])}")
        
        perf_info["analysis"] = " | ".join(analysis_parts) if analysis_parts else "Metrics collected"
        
        # Cleanup
        os.unlink(profile_script)
        if os.path.exists(db_path):
            os.unlink(db_path)
            
    except subprocess.TimeoutExpired:
        perf_info["analysis"] = "rocprofv3 timed out"
    except FileNotFoundError:
        perf_info["analysis"] = "rocprofv3 not found"
    except Exception as e:
        perf_info["analysis"] = f"rocprofv3 error: {str(e)[:200]}"
    
    return perf_info


def evaluate(problem_path: str, code_path: str, run_profiler: bool = False, backend: str = "hip") -> dict:
    """Evaluate generated code against reference.
    
    Args:
        problem_path: Path to KernelBench problem file
        code_path: Path to generated code file
        run_profiler: Whether to run rocprof analysis
        backend: Backend type ('hip' or 'triton')
    """
    result = {
        "problem": Path(problem_path).stem,
        "code_path": code_path,
        "backend": backend,
        "compile_success": False,
        "accuracy_pass": False,
        "max_diff": float('inf'),
        "mean_diff": float('inf'),
        "has_nan": True,
        "has_inf": True,
        "ref_time_ms": 0.0,
        "new_time_ms": 0.0,
        "speedup": 0.0,
        "perf_analysis": "",
        "error": None
    }
    
    try:
        # Load reference
        print(f"Loading problem: {problem_path}")
        ref_module = load_problem_module(problem_path)
        
        Model = ref_module.get('Model')
        get_inputs = ref_module.get('get_inputs')
        get_init_inputs = ref_module.get('get_init_inputs')
        
        if not Model or not get_inputs:
            result["error"] = "Problem file missing Model or get_inputs"
            return result
        
        # Load generated code
        print(f"Loading generated code: {code_path} (backend={backend})")
        try:
            gen_module = load_generated_code(code_path, backend=backend)
            result["compile_success"] = True
        except Exception as e:
            # Capture full traceback for compile errors - this contains actual compiler messages
            full_error = traceback.format_exc()
            # Extract the most relevant part (last 3000 chars usually contain compiler output)
            if len(full_error) > 3000:
                error_excerpt = "..." + full_error[-3000:]
            else:
                error_excerpt = full_error
            result["error"] = f"Compile error: {error_excerpt}"
            return result
        
        ModelNew = gen_module.get('ModelNew')
        if not ModelNew:
            result["error"] = "Generated code missing ModelNew class"
            return result
        
        # Create models
        print("Creating models...")
        torch.manual_seed(42)
        
        init_inputs = get_init_inputs() if get_init_inputs else []
        
        # Handle different model initialization patterns
        if init_inputs:
            ref_model = Model(*init_inputs).cuda()
            new_model = ModelNew(*init_inputs).cuda()
        else:
            ref_model = Model().cuda()
            new_model = ModelNew().cuda()
        
        # Get input dtype from get_inputs() to match model dtype
        torch.manual_seed(12345)
        sample_inputs = get_inputs()
        input_dtype = None
        for inp in sample_inputs:
            if isinstance(inp, torch.Tensor) and inp.is_floating_point():
                input_dtype = inp.dtype
                break
        
        # Convert models to input dtype (critical for bf16/fp16 inputs)
        if input_dtype is not None:
            ref_model = ref_model.to(input_dtype)
            new_model = new_model.to(input_dtype)
        
        # Copy weights from reference to new model (handle different naming conventions)
        ref_state = ref_model.state_dict()
        new_state = new_model.state_dict()
        
        # First, try direct key matching
        for key in ref_state:
            if key in new_state and ref_state[key].shape == new_state[key].shape:
                new_state[key] = ref_state[key].clone()
        
        # Second, try shape-based matching for unmatched weights
        ref_unmatched = {k: v for k, v in ref_state.items() if k not in new_state}
        new_unmatched = {k: v for k, v in new_state.items() if k not in ref_state}
        
        for ref_key, ref_val in ref_unmatched.items():
            for new_key, new_val in new_unmatched.items():
                if ref_val.shape == new_val.shape:
                    new_state[new_key] = ref_val.clone()
                    break
        
        new_model.load_state_dict(new_state, strict=False)
        
        # Get inputs (use same seed for reproducibility)
        torch.manual_seed(12345)
        inputs = get_inputs()
        inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
        
        # Run models with THE SAME inputs
        print("Running correctness test...")
        with torch.no_grad():
            ref_output = ref_model(*inputs)
            try:
                # Same inputs for new model
                new_output = new_model(*inputs)
            except Exception as e:
                result["error"] = f"Runtime error: {str(e)}"
                return result
        
        # Check correctness
        if isinstance(ref_output, tuple):
            ref_output = ref_output[0]
        if isinstance(new_output, tuple):
            new_output = new_output[0]
        
        # Convert to same dtype for comparison
        ref_output = ref_output.float()
        new_output = new_output.float()
        
        diff = (ref_output - new_output).abs()
        result["max_diff"] = diff.max().item()
        result["mean_diff"] = diff.mean().item()
        result["has_nan"] = torch.isnan(new_output).any().item()
        result["has_inf"] = torch.isinf(new_output).any().item()
        
        # Accuracy check - use relative tolerance for large values
        # For float16/bf16 matmul, numerical errors can accumulate significantly
        # Large matrix multiplications (K > 1000) can have 3-5% relative error
        ref_abs_max = ref_output.abs().max().item()
        # Use 5% relative tolerance for bf16, which is standard for half precision
        relative_tolerance = max(1.0, ref_abs_max * 0.05)  # 5% relative or 1.0 absolute
        
        result["accuracy_pass"] = (
            not result["has_nan"] and 
            not result["has_inf"] and 
            result["max_diff"] < relative_tolerance
        )
        
        if not result["accuracy_pass"] and result["max_diff"] < relative_tolerance * 2:
            print(f"Note: tolerance={relative_tolerance:.2f}, ref_max={ref_abs_max:.2f}")
        
        print(f"Max diff: {result['max_diff']:.6f}")
        print(f"Accuracy: {'PASS' if result['accuracy_pass'] else 'FAIL'}")
        
        # Benchmark
        if result["accuracy_pass"]:
            print("Running benchmark...")
            with torch.no_grad():
                result["ref_time_ms"] = benchmark(lambda: ref_model(*inputs))
                result["new_time_ms"] = benchmark(lambda: new_model(*inputs))
            
            result["speedup"] = result["ref_time_ms"] / result["new_time_ms"] if result["new_time_ms"] > 0 else 0
            
            print(f"Reference: {result['ref_time_ms']:.3f} ms")
            print(f"New: {result['new_time_ms']:.3f} ms")
            print(f"Speedup: {result['speedup']:.2f}x")
            
            # Run rocprof analysis for performance insights
            if run_profiler:
                print("Running rocprof analysis...")
                try:
                    perf_info = run_rocprof_analysis(code_path, problem_path)
                    result["perf_analysis"] = perf_info.get("analysis", "")
                    # Store detailed metrics for optimization
                    result["rocprof_metrics"] = {
                        "kernel_name": perf_info.get("kernel_name"),
                        "duration_ms": perf_info.get("duration_ms", 0),
                        "l2_cache_hit_rate": perf_info.get("l2_cache_hit_rate", 0),
                        "mfma_utilization_pct": perf_info.get("mfma_utilization_pct", 0),
                        "optimization_hints": perf_info.get("optimization_hints", []),
                        "raw_metrics": perf_info.get("raw_metrics", {})
                    }
                    if perf_info.get("analysis"):
                        print(f"Profiler: {perf_info['analysis']}")
                    if perf_info.get("optimization_hints"):
                        print(f"Optimization hints:")
                        for hint in perf_info["optimization_hints"]:
                            print(f"  - {hint}")
                except Exception as e:
                    result["perf_analysis"] = f"Profiler error: {str(e)[:100]}"
        
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated HipKittens/Triton kernel")
    parser.add_argument("--code", required=True, help="Path to generated code file")
    parser.add_argument("--problem", required=True, help="Path to KernelBench problem file")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    parser.add_argument("--profile", action="store_true", help="Run rocprof analysis for slow kernels")
    parser.add_argument("--backend", choices=BACKENDS, default="hip",
                        help="Backend type: 'hip' for HipKittens, 'triton' for Triton (default: hip)")
    args = parser.parse_args()
    
    result = evaluate(args.problem, args.code, run_profiler=args.profile, backend=args.backend)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Problem: {result['problem']}")
    print(f"Compile: {'✓' if result['compile_success'] else '✗'}")
    print(f"Accuracy: {'✓' if result['accuracy_pass'] else '✗'}")
    if result['accuracy_pass']:
        print(f"Speedup: {result['speedup']:.2f}x")
    if result['error']:
        print(f"Error: {result['error'][:200]}...")
    print("=" * 60)
    
    # Save results
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    # Return exit code
    if result['accuracy_pass'] and result['speedup'] >= 1.0:
        print("\n✓ SUCCESS: Accuracy passed and performance exceeded baseline!")
        sys.exit(0)
    elif result['accuracy_pass']:
        print(f"\n⚠ PARTIAL: Accuracy passed but speedup is {result['speedup']:.2f}x (need >= 1.0x)")
        sys.exit(1)
    else:
        print("\n✗ FAILED: Accuracy test failed")
        sys.exit(2)


if __name__ == "__main__":
    main()

