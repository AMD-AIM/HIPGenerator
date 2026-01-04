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


def load_problem_module(problem_path: str, use_importlib: bool = False):
    """Load reference model from problem file.
    
    Args:
        problem_path: Path to the problem file
        use_importlib: If True, use importlib (required if file contains @triton.jit)
    """
    # Check if file contains Triton JIT code
    with open(problem_path) as f:
        code = f.read()
    
    has_triton = '@triton' in code or 'triton.jit' in code
    
    if has_triton or use_importlib:
        # Use importlib for Triton code
        import importlib.util
        import uuid
        
        module_name = f"problem_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(module_name, problem_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {problem_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            if module_name in sys.modules:
                del sys.modules[module_name]
            raise
        
        return {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}
    else:
        # Use exec for simple PyTorch code
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


def detect_cheating(code_path: str, backend: str = "triton") -> dict:
    """Detect if generated code is cheating by using PyTorch functions instead of Triton kernels.
    
    Returns:
        dict with:
            - is_cheating: bool
            - cheating_reason: str or None
            - warnings: list of potential issues
    """
    with open(code_path, 'r') as f:
        code = f.read()
    
    result = {
        "is_cheating": False,
        "cheating_reason": None,
        "warnings": []
    }
    
    if backend != "triton":
        return result  # Only check Triton backend for now
    
    import re
    
    # =====================================================
    # CHECK 1: Forbidden PyTorch function calls in ModelNew
    # =====================================================
    forbidden_torch_patterns = [
        # Common activation/computation functions (CHEATING if used for main computation)
        (r'torch\.log_softmax\s*\(', 'torch.log_softmax() - use Triton kernel instead'),
        (r'torch\.softmax\s*\(', 'torch.softmax() - use Triton kernel instead'),
        (r'torch\.sigmoid\s*\(', 'torch.sigmoid() - use Triton kernel instead'),
        (r'torch\.tanh\s*\(', 'torch.tanh() - use Triton kernel instead'),
        (r'torch\.relu\s*\(', 'torch.relu() - use Triton kernel instead'),
        (r'torch\.gelu\s*\(', 'torch.gelu() - use Triton kernel instead'),
        (r'torch\.silu\s*\(', 'torch.silu() - use Triton kernel instead'),
        (r'torch\.mm\s*\(', 'torch.mm() - use Triton kernel instead'),
        (r'torch\.matmul\s*\(', 'torch.matmul() - use Triton kernel instead'),
        (r'torch\.bmm\s*\(', 'torch.bmm() - use Triton kernel instead'),
        (r'torch\.linear\s*\(', 'torch.linear() - use Triton kernel instead'),
        (r'torch\.exp\s*\(', 'torch.exp() - use Triton kernel instead'),
        (r'torch\.log\s*\(', 'torch.log() - use Triton kernel instead'),
        (r'torch\.sqrt\s*\(', 'torch.sqrt() - use Triton kernel instead'),
        # Functional API (also forbidden)
        (r'F\.log_softmax\s*\(', 'F.log_softmax() - use Triton kernel instead'),
        (r'F\.softmax\s*\(', 'F.softmax() - use Triton kernel instead'),
        (r'F\.sigmoid\s*\(', 'F.sigmoid() - use Triton kernel instead'),
        (r'F\.tanh\s*\(', 'F.tanh() - use Triton kernel instead'),
        (r'F\.relu\s*\(', 'F.relu() - use Triton kernel instead'),
        (r'F\.gelu\s*\(', 'F.gelu() - use Triton kernel instead'),
        (r'F\.silu\s*\(', 'F.silu() - use Triton kernel instead'),
        (r'F\.linear\s*\(', 'F.linear() - use Triton kernel instead'),
        (r'F\.layer_norm\s*\(', 'F.layer_norm() - use Triton kernel instead'),
        (r'F\.batch_norm\s*\(', 'F.batch_norm() - use Triton kernel instead'),
        # nn modules called directly
        (r'torch\.nn\.functional\.', 'torch.nn.functional.* - use Triton kernel instead'),
    ]
    
    # Extract ModelNew class code ONLY (not Model class which is the reference)
    # Look for class ModelNew and capture until next class or end of file
    modelnew_match = re.search(r'class\s+ModelNew[^:]*:\s*\n(.*?)(?=\nclass\s|\Z)', code, re.DOTALL)
    if modelnew_match:
        modelnew_code = modelnew_match.group(0)
        
        # Check for forbidden patterns only in ModelNew class
        for pattern, msg in forbidden_torch_patterns:
            if re.search(pattern, modelnew_code, re.IGNORECASE):
                result["is_cheating"] = True
                result["cheating_reason"] = f"CHEATING DETECTED: ModelNew uses {msg}"
                return result
    
    # If no ModelNew class found, skip this check (will fail later anyway)
    
    # =====================================================
    # CHECK 2: Malformed @triton.autotune decorator
    # =====================================================
    # Check for @triton.autotune without proper closing )
    autotune_blocks = re.findall(r'@triton\.autotune\s*\((.*?)(?=@triton\.jit|def\s+\w+)', code, re.DOTALL)
    for block in autotune_blocks:
        # Count parentheses - should be balanced
        open_parens = block.count('(')
        close_parens = block.count(')')
        if open_parens > close_parens:
            result["is_cheating"] = True
            result["cheating_reason"] = "CHEATING DETECTED: @triton.autotune decorator is malformed (missing closing parenthesis)"
            return result
    
    # =====================================================
    # CHECK 3: Kernel function definition without proper ):
    # =====================================================
    # Look for @triton.jit followed by def without ): ending the parameter list
    jit_sections = re.split(r'@triton\.jit', code)[1:]  # Skip everything before first @triton.jit
    for section in jit_sections:
        # Find the def statement immediately following @triton.jit
        def_match = re.match(r'\s*\ndef\s+(\w+)\s*\(', section)
        if def_match:
            kernel_name = def_match.group(1)
            
            # CRITICAL: Check if the KERNEL's parameter list closes BEFORE class definition
            # This catches cases where the function definition is malformed:
            # def kernel(       <- @triton.jit decorated
            #     param,
            #     code_line     <- should be in body, not parameters!
            # class ModelNew:   <- class starts before ): found
            
            # Find where ): occurs (parameter list end)
            param_close_match = re.search(r'\)\s*:', section[:1000])  # Look in first 1000 chars
            # Find where class definition starts
            class_start_match = re.search(r'\nclass\s+', section)
            
            if param_close_match:
                param_close_pos = param_close_match.start()
            else:
                param_close_pos = float('inf')
            
            if class_start_match:
                class_start_pos = class_start_match.start()
            else:
                class_start_pos = float('inf')
            
            # If class appears BEFORE the parameter list closes, the kernel is malformed
            if class_start_pos < param_close_pos:
                result["is_cheating"] = True
                result["cheating_reason"] = f"CHEATING DETECTED: Kernel '{kernel_name}' has malformed definition - 'class' appears before '):', indicating code was put in parameter list instead of body"
                return result
            
            # Also check: if no ): found at all in a reasonable range, it's malformed
            if param_close_pos == float('inf'):
                result["is_cheating"] = True
                result["cheating_reason"] = f"CHEATING DETECTED: Kernel '{kernel_name}' has malformed function definition (missing '):')"
                return result
    
    # =====================================================
    # CHECK 4: Empty or incomplete Triton kernels
    # =====================================================
    # Find all properly defined kernels
    kernel_pattern = r'@triton\.jit\s*\ndef\s+(\w+)\s*\([^)]*\)\s*:\s*\n(.*?)(?=\n@|\nclass\s|\ndef\s|\Z)'
    kernels = re.findall(kernel_pattern, code, re.DOTALL)
    
    # Also count @triton.jit occurrences
    triton_jit_count = len(re.findall(r'@triton\.jit', code))
    
    # If we have @triton.jit but couldn't parse any complete kernels, that's suspicious
    if triton_jit_count > 0 and len(kernels) == 0:
        result["is_cheating"] = True
        result["cheating_reason"] = "CHEATING DETECTED: @triton.jit found but no complete kernel definitions (check for syntax errors)"
        return result
    
    for kernel_name, kernel_body in kernels:
        # Check if kernel body is essentially empty or just has return
        stripped_body = kernel_body.strip()
        lines = [l.strip() for l in stripped_body.split('\n') if l.strip() and not l.strip().startswith('#')]
        
        # Remove comments from kernel body for accurate checking
        # This prevents false positives from comments like "# Missing: tl.store(...)"
        code_without_comments = '\n'.join(
            line.split('#')[0] for line in kernel_body.split('\n')
        )
        
        # Count meaningful operations (excluding comments)
        has_tl_load = 'tl.load' in code_without_comments
        has_tl_store = 'tl.store' in code_without_comments
        
        if len(lines) < 3:  # Too short to be a real kernel
            result["is_cheating"] = True
            result["cheating_reason"] = f"CHEATING DETECTED: Kernel '{kernel_name}' has incomplete implementation (< 3 lines)"
            return result
        
        # CRITICAL: A real kernel must have both tl.load AND tl.store
        if not has_tl_load or not has_tl_store:
            result["is_cheating"] = True
            result["cheating_reason"] = f"CHEATING DETECTED: Kernel '{kernel_name}' is incomplete - missing {'tl.load' if not has_tl_load else 'tl.store'}"
            return result
    
    # =====================================================
    # CHECK 5: No Triton kernels defined at all
    # =====================================================
    if '@triton.jit' not in code and backend == "triton":
        result["is_cheating"] = True
        result["cheating_reason"] = "CHEATING DETECTED: No @triton.jit kernel defined - ModelNew must use Triton kernels"
        return result
    
    # =====================================================
    # CHECK 5: Fallback patterns in ModelNew.forward()
    # =====================================================
    # Check for else/return patterns that fallback to PyTorch
    fallback_patterns = [
        r'else:\s*\n\s*return\s+torch\.',  # else: return torch.xxx
        r'return\s+torch\.nn\.functional\.',  # return F.xxx
        r'except.*:\s*\n\s*return\s+torch\.',  # except: return torch.xxx
    ]
    
    if modelnew_match:
        for pattern in fallback_patterns:
            if re.search(pattern, modelnew_code):
                result["is_cheating"] = True
                result["cheating_reason"] = "CHEATING DETECTED: ModelNew has PyTorch fallback in forward() - all cases must use Triton"
                return result
    
    return result


def benchmark(fn, warmup=30, iterations=100):
    """Benchmark a function with extended warmup for autotune stability."""
    # Extended warmup to let autotune settle
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    
    # Additional warmup after autotune
    for _ in range(10):
        _ = fn()
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        _ = fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iterations * 1000  # ms


def run_rocprof_analysis(code_path: str, problem_path: str, backend: str = "hip") -> dict:
    """Run rocprofv3 to get detailed kernel performance metrics with PMC counters.
    
    Collects:
    - Per-kernel PMC counters (VALU, SALU, LDS conflicts, L2 hit/miss)
    - Resource usage (VGPR, SGPR, LDS)
    - Grid/workgroup configuration
    """
    perf_info = {
        "kernel_name": None,
        "duration_us": 0.0,
        "duration_ms": 0.0,
        "total_calls": 0,
        "avg_duration_us": 0.0,
        "percentage": 0.0,
        # Resource usage
        "vgpr_count": 0,
        "sgpr_count": 0,
        "lds_usage_bytes": 0,
        # Grid/block config
        "grid_size": [],
        "workgroup_size": [],
        # PMC counters
        "pmc_counters": {},
        # Raw metrics for analysis
        "raw_metrics": {},
        "analysis": "",
        "optimization_hints": []
    }
    
    # Essential PMC counters
    pmc_counters = "SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_SALU,SQ_LDS_BANK_CONFLICT,TCC_HIT,TCC_MISS"
    
    try:
        import sqlite3
        
        # Create profiling script
        if backend == "triton":
            profile_code = f'''
import os
import sys
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch
import torch.nn as nn
import importlib.util

# Load problem
exec(open("{problem_path}").read())

# Load Triton code via importlib
spec = importlib.util.spec_from_file_location("triton_module", "{code_path}")
triton_module = importlib.util.module_from_spec(spec)
sys.modules["triton_module"] = triton_module
spec.loader.exec_module(triton_module)
ModelNew = triton_module.ModelNew

torch.manual_seed(42)
init_inputs = get_init_inputs() if 'get_init_inputs' in dir() else []
model = ModelNew(*init_inputs).cuda() if init_inputs else ModelNew().cuda()

inputs = get_inputs()
inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

with torch.no_grad():
    for _ in range(5):
        _ = model(*inputs)
    torch.cuda.synchronize()
    for _ in range(10):
        _ = model(*inputs)
    torch.cuda.synchronize()
'''
        else:
            profile_code = f'''
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
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(profile_code)
            profile_script = f.name
        
        db_path = tempfile.mktemp(suffix='_results.db')
        output_prefix = db_path.replace('_results.db', '')
        
        # Run rocprofv3 with PMC counters and kernel trace
        cmd = [
            'rocprofv3',
            '--pmc', pmc_counters,
            '--kernel-trace', 
            '-o', output_prefix,
            '--',
            'python3', profile_script
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=os.environ
        )
        
        # Parse SQLite database
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Query counters_collection for detailed metrics
            try:
                cursor.execute("""
                    SELECT 
                        kernel_name, counter_name, SUM(value) as total,
                        MAX(vgpr_count), MAX(sgpr_count), MAX(lds_block_size),
                        MAX(workgroup_size), MAX(grid_size), AVG(duration)
                    FROM counters_collection
                    WHERE kernel_name NOT LIKE '%at::native%'
                      AND kernel_name NOT LIKE '%distribution%'
                      AND kernel_name NOT LIKE '%elementwise_kernel%'
                    GROUP BY kernel_name, counter_name
                """)
                rows = cursor.fetchall()
                
                kernel_counters = {}
                for row in rows:
                    kernel_name = row[0]
                    counter_name = row[1]
                    
                    if kernel_name not in kernel_counters:
                        kernel_counters[kernel_name] = {
                            "vgpr": row[3] or 0,
                            "sgpr": row[4] or 0,
                            "lds_bytes": row[5] or 0,
                            "workgroup_size": row[6] or 0,
                            "grid_size": row[7] or 0,
                            "avg_duration_ns": row[8] or 0,
                            "counters": {}
                        }
                    
                    kernel_counters[kernel_name]["counters"][counter_name] = row[2] or 0
                
                # Find our Triton kernel (not PyTorch internals)
                for kname, kdata in kernel_counters.items():
                    perf_info["kernel_name"] = kname[:80]
                    perf_info["vgpr_count"] = kdata["vgpr"]
                    perf_info["sgpr_count"] = kdata["sgpr"]
                    perf_info["lds_usage_bytes"] = kdata["lds_bytes"]
                    perf_info["workgroup_size"] = [kdata["workgroup_size"], 1, 1]
                    perf_info["grid_size"] = [kdata["grid_size"], 1, 1]
                    perf_info["avg_duration_us"] = kdata["avg_duration_ns"] / 1000.0
                    perf_info["pmc_counters"] = kdata["counters"]
                    break
                    
            except Exception as e:
                perf_info["raw_metrics"]["pmc_error"] = str(e)
            
            # Also query kernel_summary for timing stats
            try:
                cursor.execute("""
                    SELECT name, calls, "DURATION(nsec)", "AVERAGE(nsec)", "PERCENT"
                    FROM kernel_summary
                    WHERE name NOT LIKE '%at::native%'
                    LIMIT 5
                """)
                summary_rows = cursor.fetchall()
                
                for row in summary_rows:
                    if row[0] == perf_info.get("kernel_name"):
                        perf_info["total_calls"] = row[1] or 0
                        perf_info["duration_us"] = (row[2] or 0) / 1000.0
                        perf_info["percentage"] = row[4] or 0
                        break
            except:
                pass
            
            conn.close()
        
        # Generate optimization hints from PMC counters
        hints = []
        counters = perf_info.get("pmc_counters", {})
        
        # Check L2 cache hit rate
        tcc_hit = counters.get("TCC_HIT", 0)
        tcc_miss = counters.get("TCC_MISS", 0)
        tcc_total = tcc_hit + tcc_miss
        if tcc_total > 0:
            l2_hit_rate = tcc_hit / tcc_total * 100
            perf_info["raw_metrics"]["l2_hit_rate"] = l2_hit_rate
            if l2_hit_rate < 50:
                hints.append(f"LOW L2 HIT RATE ({l2_hit_rate:.1f}%): Consider tile grouping for better cache reuse")
        
        # Check LDS bank conflicts
        lds_conflicts = counters.get("SQ_LDS_BANK_CONFLICT", 0)
        waves = counters.get("SQ_WAVES", 0)
        if waves > 0 and lds_conflicts / waves > 1:
            hints.append(f"LDS BANK CONFLICTS ({lds_conflicts/waves:.1f}/wave): Optimize LDS access pattern")
        
        # Check VGPR usage
        vgpr = perf_info.get("vgpr_count", 0)
        if vgpr > 128:
            hints.append(f"HIGH VGPR ({vgpr}): May limit occupancy, reduce register pressure")
        
        # Check workgroup size
        wg_size = perf_info.get("workgroup_size", [0])[0]
        if wg_size > 0 and wg_size < 256:
            hints.append(f"SMALL WORKGROUP ({wg_size}): Consider larger blocks")
        
        # Check instruction mix
        valu = counters.get("SQ_INSTS_VALU", 0)
        salu = counters.get("SQ_INSTS_SALU", 0)
        if salu > valu * 2:
            hints.append(f"HIGH SALU/VALU RATIO ({salu/valu:.1f}x): Too much scalar work")
        
        # Check kernel time percentage
        if perf_info["percentage"] < 80:
            hints.append(f"KERNEL ONLY {perf_info['percentage']:.1f}% of time: Check for host-side overhead")
        
        perf_info["optimization_hints"] = hints
        
        # Build analysis summary
        analysis_parts = []
        if perf_info["kernel_name"]:
            name_short = perf_info["kernel_name"].split('(')[0][:30]
            analysis_parts.append(f"Kernel: {name_short}")
        if perf_info["avg_duration_us"] > 0:
            analysis_parts.append(f"Avg: {perf_info['avg_duration_us']:.1f}us")
        if perf_info["vgpr_count"] > 0:
            analysis_parts.append(f"VGPR: {perf_info['vgpr_count']}")
        if perf_info["lds_usage_bytes"] > 0:
            analysis_parts.append(f"LDS: {perf_info['lds_usage_bytes']/1024:.1f}KB")
        if "l2_hit_rate" in perf_info["raw_metrics"]:
            analysis_parts.append(f"L2: {perf_info['raw_metrics']['l2_hit_rate']:.1f}%")
        
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


def analyze_kernel_code(code_path: str) -> dict:
    """Analyze Triton kernel code to identify optimization opportunities."""
    with open(code_path, 'r') as f:
        code = f.read()
    
    analysis = {
        "has_autotune": "@triton.autotune" in code,
        "has_xcd_swizzle": "NUM_XCDS" in code or "xcd_id" in code,
        "has_group_m": "GROUP_M" in code or "group_id" in code,
        "has_env_vars": "TRITON_HIP_USE_BLOCK_PINGPONG" in code,
        "has_precompute": "register_buffer" in code or "self._grid" in code or "self.grid" in code,
        "block_sizes": [],
        "num_warps": None,
        "num_stages": None,
        "is_elementwise": False,
        "is_gemm": False,
        "has_internal_loop": False,
        "issues": [],
        "suggestions": []
    }
    
    import re
    
    # Detect operation type
    code_lower = code.lower()
    analysis["is_gemm"] = any(op in code_lower for op in ['matmul', 'mm(', 'bmm(', 'linear', 'gemm', 'dot('])
    analysis["is_elementwise"] = any(op in code_lower for op in [
        'sigmoid', 'relu', 'gelu', 'tanh', 'swish', 'silu', 'softmax', 'exp(', 'log('
    ]) and not analysis["is_gemm"]
    
    # Extract block sizes from autotune or constexpr
    block_matches = re.findall(r"BLOCK_(?:SIZE|M|N|K)['\"]?\s*[:\=]\s*(\d+)", code)
    analysis["block_sizes"] = [int(b) for b in block_matches]
    
    # Check for num_warps
    warps_match = re.search(r"num_warps\s*=\s*(\d+)", code)
    if warps_match:
        analysis["num_warps"] = int(warps_match.group(1))
    
    # Check for internal loops in element-wise kernels (BAD!)
    if analysis["is_elementwise"]:
        if re.search(r'for\s+\w+\s+in\s+range\s*\(.*BLOCK', code):
            analysis["has_internal_loop"] = True
            analysis["issues"].append("CRITICAL: Internal loop detected in element-wise kernel - this destroys performance!")
            analysis["suggestions"].append("Remove internal loops - process entire BLOCK_SIZE at once with tl.arange()")
    
    # Generate suggestions based on analysis
    if not analysis["has_autotune"]:
        analysis["issues"].append("Missing @triton.autotune decorator")
        if analysis["is_gemm"]:
            analysis["suggestions"].append("""Add @triton.autotune with GEMM configs:
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)""")
        else:
            analysis["suggestions"].append("""Add @triton.autotune with element-wise configs:
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)""")
    
    if analysis["is_gemm"]:
        if not analysis["has_xcd_swizzle"]:
            analysis["issues"].append("Missing XCD swizzle for GEMM (32 XCDs on MI350)")
            analysis["suggestions"].append("Add XCD swizzle: pid % 32 for xcd_id, pid // 32 for local_pid")
        
        if not analysis["has_group_m"]:
            analysis["issues"].append("Missing L2 cache grouping (GROUP_M)")
            analysis["suggestions"].append("Add GROUP_M=8 parameter and tile grouping for better L2 hit rate")
    
    if not analysis["has_env_vars"]:
        analysis["issues"].append("Missing MI350 environment variables")
        analysis["suggestions"].append("Add: os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'")
    
    # Check block sizes
    if analysis["block_sizes"]:
        max_block = max(analysis["block_sizes"])
        if analysis["is_elementwise"] and max_block < 1024:
            analysis["issues"].append(f"Small BLOCK_SIZE ({max_block}) for element-wise op")
            analysis["suggestions"].append("Use larger BLOCK_SIZE (2048-8192) for better memory throughput")
    
    return analysis


def build_profiler_feedback(eval_result: dict, perf_info: dict, code_path: str = None) -> str:
    """Build detailed feedback from profiler results for optimization.
    
    Combines runtime profiler metrics with static code analysis to provide
    targeted optimization suggestions.
    """
    feedback_parts = []
    
    # Performance summary
    feedback_parts.append("=" * 60)
    feedback_parts.append("KERNEL PERFORMANCE ANALYSIS")
    feedback_parts.append("=" * 60)
    
    baseline_time = eval_result.get('baseline_time_ms') or eval_result.get('ref_time_ms', 0)
    optimized_time = eval_result.get('optimized_time_ms') or eval_result.get('new_time_ms', 0)
    speedup = eval_result.get('speedup', 0)
    
    feedback_parts.append(f"\nBaseline Time: {baseline_time:.3f} ms")
    feedback_parts.append(f"Current Time:  {optimized_time:.3f} ms")
    feedback_parts.append(f"Speedup:       {speedup:.2f}x {'✓' if speedup >= 1.0 else '⚠ NEEDS IMPROVEMENT'}")
    
    # Profiler metrics
    if perf_info.get("kernel_name"):
        feedback_parts.append(f"\n--- Runtime Profiler Metrics ---")
        feedback_parts.append(f"Kernel: {perf_info['kernel_name'][:80]}")
        
    if perf_info.get("avg_duration_us", 0) > 0:
        feedback_parts.append(f"Avg kernel time: {perf_info['avg_duration_us']:.1f} μs")
    
    if perf_info.get("lds_usage_bytes", 0) > 0:
        lds_kb = perf_info['lds_usage_bytes'] / 1024
        lds_status = "⚠ HIGH (may limit occupancy)" if lds_kb > 48 else "OK"
        feedback_parts.append(f"LDS Usage: {lds_kb:.1f} KB [{lds_status}]")
    
    if perf_info.get("workgroup_size"):
        wg = perf_info['workgroup_size']
        total_threads = wg[0] * wg[1] * wg[2] if len(wg) == 3 else 0
        wg_status = "⚠ SMALL" if total_threads < 256 else "OK"
        feedback_parts.append(f"Workgroup: {wg[0]}x{wg[1]}x{wg[2]} = {total_threads} threads [{wg_status}]")
    
    if perf_info.get("grid_size"):
        grid = perf_info['grid_size']
        total_blocks = grid[0] * grid[1] * grid[2] if len(grid) == 3 else grid[0]
        feedback_parts.append(f"Grid: {grid[0]}x{grid[1]}x{grid[2]} = {total_blocks} blocks")
        # Check if enough blocks for all XCDs
        if total_blocks < 256:
            feedback_parts.append(f"  ⚠ Only {total_blocks} blocks - may not fully utilize 32 XCDs (256 CUs)")
    
    # Static code analysis
    code_analysis = None
    if code_path:
        try:
            code_analysis = analyze_kernel_code(code_path)
            
            feedback_parts.append(f"\n--- Code Analysis ---")
            feedback_parts.append(f"Operation type: {'GEMM/Matmul' if code_analysis['is_gemm'] else 'Element-wise' if code_analysis['is_elementwise'] else 'Unknown'}")
            feedback_parts.append(f"Has @triton.autotune: {'✓' if code_analysis['has_autotune'] else '✗ MISSING'}")
            
            if code_analysis['is_gemm']:
                feedback_parts.append(f"Has XCD swizzle: {'✓' if code_analysis['has_xcd_swizzle'] else '✗ MISSING'}")
                feedback_parts.append(f"Has L2 grouping (GROUP_M): {'✓' if code_analysis['has_group_m'] else '✗ MISSING'}")
            
            feedback_parts.append(f"Has env vars: {'✓' if code_analysis['has_env_vars'] else '✗ MISSING'}")
            
            if code_analysis['block_sizes']:
                feedback_parts.append(f"Block sizes found: {code_analysis['block_sizes']}")
            
            if code_analysis['has_internal_loop']:
                feedback_parts.append(f"⚠ CRITICAL: Has internal loop in element-wise kernel!")
                
        except Exception as e:
            feedback_parts.append(f"Code analysis error: {str(e)[:100]}")
    
    # Generate targeted optimization suggestions
    feedback_parts.append(f"\n{'=' * 60}")
    feedback_parts.append("OPTIMIZATION RECOMMENDATIONS")
    feedback_parts.append("=" * 60)
    
    recommendations = []
    priority = 1
    
    # Priority 1: Add autotune if missing
    if code_analysis and not code_analysis['has_autotune']:
        recommendations.append(f"""
{priority}. [HIGH PRIORITY] Add @triton.autotune decorator
   WHY: Autotune automatically finds optimal block sizes and num_warps
   HOW: {code_analysis['suggestions'][0] if code_analysis['suggestions'] else 'Add @triton.autotune with multiple configs'}
""")
        priority += 1
    
    # Priority 2: Fix internal loops for element-wise
    if code_analysis and code_analysis['has_internal_loop']:
        recommendations.append(f"""
{priority}. [CRITICAL] Remove internal loops in element-wise kernel
   WHY: Internal loops destroy parallelism and memory throughput
   HOW: Process entire BLOCK_SIZE at once:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets, mask=mask)
        y = compute(x)
        tl.store(output_ptr + offsets, y, mask=mask)
""")
        priority += 1
    
    # Priority 3: XCD swizzle for GEMM
    if code_analysis and code_analysis['is_gemm'] and not code_analysis['has_xcd_swizzle']:
        recommendations.append(f"""
{priority}. [HIGH PRIORITY] Add XCD swizzle for MI350 (32 XCDs)
   WHY: Without XCD swizzle, only ~3% of GPU utilized (1/32 XCDs)
   HOW: 
        NUM_XCDS = 32
        pids_per_xcd = tl.cdiv(num_pids, NUM_XCDS)
        xcd_id = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        pid = xcd_id * pids_per_xcd + local_pid
""")
        priority += 1
    
    # Priority 4: L2 cache grouping for GEMM
    if code_analysis and code_analysis['is_gemm'] and not code_analysis['has_group_m']:
        recommendations.append(f"""
{priority}. [MEDIUM PRIORITY] Add L2 cache grouping (GROUP_M)
   WHY: Groups adjacent tiles to improve L2 cache hit rate
   HOW:
        GROUP_M = 8
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
""")
        priority += 1
    
    # Priority 5: Environment variables
    if code_analysis and not code_analysis['has_env_vars']:
        recommendations.append(f"""
{priority}. [MEDIUM PRIORITY] Add MI350 environment variables
   WHY: Enables advanced memory prefetching and async operations
   HOW: Add at top of file:
        import os
        os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
        os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'
""")
        priority += 1
    
    # Priority 6: Block size optimization
    if code_analysis and code_analysis['is_elementwise'] and code_analysis['block_sizes']:
        max_block = max(code_analysis['block_sizes']) if code_analysis['block_sizes'] else 0
        if max_block < 2048:
            recommendations.append(f"""
{priority}. [MEDIUM PRIORITY] Increase BLOCK_SIZE for element-wise ops
   WHY: Larger blocks improve memory throughput for bandwidth-bound ops
   HOW: Use BLOCK_SIZE in range 2048-8192, let autotune pick best
""")
            priority += 1
    
    # Add recommendations to feedback
    if recommendations:
        for rec in recommendations:
            feedback_parts.append(rec)
    else:
        feedback_parts.append("\n✓ Code looks well-optimized. Consider fine-tuning autotune configs.")
    
    return "\n".join(feedback_parts)


def evaluate(problem_path: str, code_path: str, run_profiler: bool = False, backend: str = "hip",
             ref_class_name: str = "Model", new_class_name: str = "ModelNew") -> dict:
    """Evaluate generated code against reference.
    
    Args:
        problem_path: Path to KernelBench problem file
        code_path: Path to generated code file
        run_profiler: Whether to run rocprof analysis
        backend: Backend type ('hip' or 'triton')
        ref_class_name: Name of the reference model class (default: Model)
        new_class_name: Name of the new/generated model class (default: ModelNew)
    """
    result = {
        "problem": Path(problem_path).stem,
        "code_path": code_path,
        "backend": backend,
        "ref_class_name": ref_class_name,
        "new_class_name": new_class_name,
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
        "profiler_feedback": "",
        "error": None
    }
    
    try:
        # Check for cheating FIRST (before any loading) - this works on raw file content
        print(f"Checking for cheating patterns in: {code_path}")
        cheat_check = detect_cheating(code_path, backend=backend)
        if cheat_check["is_cheating"]:
            result["error"] = cheat_check["cheating_reason"]
            result["compile_success"] = False
            result["accuracy_pass"] = False
            print(f"❌ {cheat_check['cheating_reason']}")
            return result
        
        if cheat_check["warnings"]:
            for warning in cheat_check["warnings"]:
                print(f"⚠️ Warning: {warning}")
        
        # Load reference (use separate file if problem_path != code_path)
        print(f"Loading problem: {problem_path}")
        ref_module = load_problem_module(problem_path)
        
        # Try to find the reference class by name
        Model = ref_module.get(ref_class_name)
        get_inputs = ref_module.get('get_inputs')
        get_init_inputs = ref_module.get('get_init_inputs')
        
        if not Model:
            # Fallback: try 'Model' as default
            Model = ref_module.get('Model')
        
        if not Model or not get_inputs:
            result["error"] = f"Problem file missing {ref_class_name}/Model or get_inputs"
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
        
        # Try to find the new class by name
        ModelNew = gen_module.get(new_class_name)
        if not ModelNew:
            # Fallback: try 'ModelNew' as default
            ModelNew = gen_module.get('ModelNew')
        if not ModelNew:
            result["error"] = f"Generated code missing {new_class_name}/ModelNew class"
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
                    perf_info = run_rocprof_analysis(code_path, problem_path, backend=backend)
                    result["perf_analysis"] = perf_info.get("analysis", "")
                    # Store detailed metrics for optimization
                    result["rocprof_metrics"] = {
                        "kernel_name": perf_info.get("kernel_name"),
                        "duration_ms": perf_info.get("duration_ms", 0),
                        "avg_duration_us": perf_info.get("avg_duration_us", 0),
                        "total_calls": perf_info.get("total_calls", 0),
                        "lds_usage_bytes": perf_info.get("lds_usage_bytes", 0),
                        "workgroup_size": perf_info.get("workgroup_size", []),
                        "grid_size": perf_info.get("grid_size", []),
                        "optimization_hints": perf_info.get("optimization_hints", []),
                        "raw_metrics": perf_info.get("raw_metrics", {})
                    }
                    if perf_info.get("analysis"):
                        print(f"Profiler: {perf_info['analysis']}")
                    if perf_info.get("optimization_hints"):
                        print(f"Optimization hints:")
                        for hint in perf_info["optimization_hints"]:
                            print(f"  - {hint}")
                    
                    # Build profiler feedback for optimization (pass code_path for static analysis)
                    result["profiler_feedback"] = build_profiler_feedback(result, perf_info, code_path)
                except Exception as e:
                    result["perf_analysis"] = f"Profiler error: {str(e)[:100]}"
        
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
    
    return result


def evaluate_triton_optimize(baseline_code_path: str, optimized_code_path: str, 
                              baseline_class: str, optimized_class: str,
                              run_profiler: bool = False) -> dict:
    """Evaluate optimized Triton kernel against user's baseline Triton kernel.
    
    For triton2triton optimization:
    - User's Triton kernel is the BASELINE (both for accuracy and performance)
    - Optimized kernel must match baseline's output AND be faster
    
    Args:
        baseline_code_path: Path to user's original Triton code (contains baseline class)
        optimized_code_path: Path to optimized Triton code (contains optimized class)
        baseline_class: Class name of user's kernel (e.g., 'ModelNew')
        optimized_class: Class name of optimized kernel (e.g., 'ModelNewOptimized')
        run_profiler: Whether to run rocprof analysis
    """
    result = {
        "problem": "triton_optimize",
        "baseline_code_path": baseline_code_path,
        "optimized_code_path": optimized_code_path,
        "baseline_class": baseline_class,
        "optimized_class": optimized_class,
        "compile_success": False,
        "accuracy_pass": False,
        "max_diff": float('inf'),
        "mean_diff": float('inf'),
        "has_nan": True,
        "has_inf": True,
        "baseline_time_ms": 0.0,
        "optimized_time_ms": 0.0,
        "speedup": 0.0,
        "perf_analysis": "",
        "profiler_feedback": "",
        "error": None
    }
    
    try:
        # Check for cheating in optimized code
        print(f"Checking optimized code for cheating patterns...")
        cheat_check = detect_cheating(optimized_code_path, backend="triton")
        if cheat_check["is_cheating"]:
            result["error"] = cheat_check["cheating_reason"]
            print(f"❌ {cheat_check['cheating_reason']}")
            return result
        
        # Load baseline (user's Triton kernel)
        print(f"Loading baseline: {baseline_code_path} (class: {baseline_class})")
        baseline_module = load_generated_code(baseline_code_path, backend="triton")
        BaselineModel = baseline_module.get(baseline_class)
        
        if not BaselineModel:
            result["error"] = f"Baseline file missing class '{baseline_class}'"
            return result
        
        # Get get_inputs from baseline file
        get_inputs = baseline_module.get('get_inputs')
        get_init_inputs = baseline_module.get('get_init_inputs')
        
        if not get_inputs:
            result["error"] = "Baseline file missing get_inputs() function"
            return result
        
        # Load optimized code
        print(f"Loading optimized: {optimized_code_path} (class: {optimized_class})")
        try:
            optimized_module = load_generated_code(optimized_code_path, backend="triton")
            result["compile_success"] = True
        except Exception as e:
            full_error = traceback.format_exc()
            if len(full_error) > 3000:
                error_excerpt = "..." + full_error[-3000:]
            else:
                error_excerpt = full_error
            result["error"] = f"Compile error: {error_excerpt}"
            return result
        
        OptimizedModel = optimized_module.get(optimized_class)
        if not OptimizedModel:
            result["error"] = f"Optimized code missing class '{optimized_class}'"
            return result
        
        # Create models
        print("Creating models...")
        torch.manual_seed(42)
        
        init_inputs = get_init_inputs() if get_init_inputs else []
        
        if init_inputs:
            baseline_model = BaselineModel(*init_inputs).cuda()
            optimized_model = OptimizedModel(*init_inputs).cuda()
        else:
            baseline_model = BaselineModel().cuda()
            optimized_model = OptimizedModel().cuda()
        
        # Get input dtype
        torch.manual_seed(12345)
        sample_inputs = get_inputs()
        input_dtype = None
        for inp in sample_inputs:
            if isinstance(inp, torch.Tensor) and inp.is_floating_point():
                input_dtype = inp.dtype
                break
        
        if input_dtype is not None:
            baseline_model = baseline_model.to(input_dtype)
            optimized_model = optimized_model.to(input_dtype)
        
        # Copy weights (same logic as before)
        baseline_state = baseline_model.state_dict()
        optimized_state = optimized_model.state_dict()
        
        for key in baseline_state:
            if key in optimized_state and baseline_state[key].shape == optimized_state[key].shape:
                optimized_state[key] = baseline_state[key].clone()
        
        optimized_model.load_state_dict(optimized_state, strict=False)
        
        # Get inputs
        torch.manual_seed(12345)
        inputs = get_inputs()
        inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
        
        # Run models
        print("Running correctness test (optimized vs baseline)...")
        with torch.no_grad():
            baseline_output = baseline_model(*inputs)
            try:
                optimized_output = optimized_model(*inputs)
            except Exception as e:
                result["error"] = f"Runtime error: {str(e)}"
                return result
        
        # Check correctness
        if isinstance(baseline_output, tuple):
            baseline_output = baseline_output[0]
        if isinstance(optimized_output, tuple):
            optimized_output = optimized_output[0]
        
        baseline_output = baseline_output.float()
        optimized_output = optimized_output.float()
        
        diff = (baseline_output - optimized_output).abs()
        result["max_diff"] = diff.max().item()
        result["mean_diff"] = diff.mean().item()
        result["has_nan"] = torch.isnan(optimized_output).any().item()
        result["has_inf"] = torch.isinf(optimized_output).any().item()
        
        baseline_abs_max = baseline_output.abs().max().item()
        relative_tolerance = max(1.0, baseline_abs_max * 0.05)
        
        result["accuracy_pass"] = (
            not result["has_nan"] and 
            not result["has_inf"] and 
            result["max_diff"] < relative_tolerance
        )
        
        print(f"Max diff: {result['max_diff']:.6f}")
        print(f"Accuracy: {'PASS' if result['accuracy_pass'] else 'FAIL'}")
        
        # Benchmark: compare optimized vs baseline (not vs PyTorch!)
        if result["accuracy_pass"]:
            print("Running benchmark (optimized vs baseline)...")
            with torch.no_grad():
                result["baseline_time_ms"] = benchmark(lambda: baseline_model(*inputs))
                result["optimized_time_ms"] = benchmark(lambda: optimized_model(*inputs))
            
            result["speedup"] = result["baseline_time_ms"] / result["optimized_time_ms"] if result["optimized_time_ms"] > 0 else 0
            
            # Also store as ref/new for compatibility
            result["ref_time_ms"] = result["baseline_time_ms"]
            result["new_time_ms"] = result["optimized_time_ms"]
            
            print(f"Baseline ({baseline_class}): {result['baseline_time_ms']:.3f} ms")
            print(f"Optimized ({optimized_class}): {result['optimized_time_ms']:.3f} ms")
            print(f"Speedup: {result['speedup']:.2f}x")
            
            if run_profiler:
                print("Running rocprof analysis on optimized kernel...")
                # For profiler, we need a problem file with get_inputs
                # Create a temp file that has get_inputs from baseline
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    with open(baseline_code_path, 'r') as bf:
                        baseline_code = bf.read()
                    # Extract get_inputs and get_init_inputs
                    import re
                    get_inputs_match = re.search(r'def\s+get_inputs\s*\(\s*\)[\s\S]*?(?=\ndef\s|\nclass\s|\Z)', baseline_code)
                    get_init_inputs_match = re.search(r'def\s+get_init_inputs\s*\(\s*\)[\s\S]*?(?=\ndef\s|\nclass\s|\Z)', baseline_code)
                    
                    # Build minimal problem file
                    problem_code = "import torch\nimport torch.nn as nn\n\n"
                    # Extract variable definitions (like batch_size, dim) before get_inputs
                    var_defs_match = re.search(r'((?:^[a-z_][a-z_0-9]*\s*=\s*\d+\n?)+)', baseline_code, re.MULTILINE)
                    if var_defs_match:
                        problem_code += var_defs_match.group(1) + "\n"
                    if get_inputs_match:
                        problem_code += get_inputs_match.group(0) + "\n"
                    if get_init_inputs_match:
                        problem_code += get_init_inputs_match.group(0) + "\n"
                    
                    f.write(problem_code)
                    temp_problem = f.name
                
                try:
                    perf_info = run_rocprof_analysis(optimized_code_path, temp_problem, backend="triton")
                    result["perf_analysis"] = perf_info.get("analysis", "")
                    result["rocprof_metrics"] = {
                        "kernel_name": perf_info.get("kernel_name"),
                        "avg_duration_us": perf_info.get("avg_duration_us", 0),
                        "lds_usage_bytes": perf_info.get("lds_usage_bytes", 0),
                        "workgroup_size": perf_info.get("workgroup_size", []),
                        "optimization_hints": perf_info.get("optimization_hints", []),
                    }
                    # Pass optimized_code_path for static code analysis
                    result["profiler_feedback"] = build_profiler_feedback(result, perf_info, optimized_code_path)
                except Exception as e:
                    result["perf_analysis"] = f"Profiler error: {str(e)[:100]}"
                finally:
                    os.unlink(temp_problem)
    
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
    parser.add_argument("--ref-class", default="Model", help="Reference model class name (default: Model)")
    parser.add_argument("--new-class", default="ModelNew", help="Generated model class name (default: ModelNew)")
    # New: triton2triton optimization mode
    parser.add_argument("--triton-optimize", action="store_true",
                        help="Triton2Triton mode: compare optimized kernel against user's baseline Triton kernel")
    parser.add_argument("--baseline-code", help="Path to baseline Triton code (for --triton-optimize)")
    parser.add_argument("--baseline-class", default="ModelNew", help="Baseline class name (for --triton-optimize)")
    parser.add_argument("--optimized-class", default="ModelNewOptimized", help="Optimized class name (for --triton-optimize)")
    args = parser.parse_args()
    
    # Triton2Triton optimization mode
    if args.triton_optimize:
        baseline_code = args.baseline_code or args.problem  # Use problem as baseline if not specified
        result = evaluate_triton_optimize(
            baseline_code_path=baseline_code,
            optimized_code_path=args.code,
            baseline_class=args.baseline_class,
            optimized_class=args.optimized_class,
            run_profiler=args.profile
        )
    else:
        # Standard evaluation mode
        result = evaluate(args.problem, args.code, run_profiler=args.profile, backend=args.backend,
                         ref_class_name=args.ref_class, new_class_name=args.new_class)
    
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

