#!/usr/bin/env python3
"""
Enhanced rocprofv3 profiler for Triton kernel optimization.

Collects:
- Per-kernel PMC counters (SQ_INSTS_VALU, SQ_INSTS_SALU, SQ_WAVES, TCC_HIT/MISS, etc.)
- Resource usage (VGPR, SGPR, LDS)
- Grid/workgroup configuration
- Duration statistics

Usage:
    profiler = RocprofProfiler()
    metrics = profiler.profile_kernel(code_path, problem_path)
"""

import os
import sys
import json
import sqlite3
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

# Key PMC counters for optimization feedback
# Note: rocprofv3 can only collect a limited number of counters per pass
# Use essential counters only to reduce overhead
PMC_COUNTERS = [
    "SQ_WAVES",             # Total waves
    "SQ_INSTS_VALU",        # Vector ALU instructions
    "SQ_INSTS_SALU",        # Scalar ALU instructions
    "SQ_LDS_BANK_CONFLICT", # LDS bank conflicts
    "TCC_HIT",              # L2 cache hits
    "TCC_MISS",             # L2 cache misses
]


class RocprofProfiler:
    """Enhanced rocprofv3 profiler for Triton kernels."""
    
    def __init__(self, counters: Optional[List[str]] = None, timeout: int = 300):
        self.counters = counters or PMC_COUNTERS
        self.timeout = timeout
        
    def profile_kernel(self, code_path: str, problem_path: str, 
                       warmup: int = 5, runs: int = 20) -> Dict[str, Any]:
        """Profile a Triton kernel and return detailed metrics.
        
        Args:
            code_path: Path to generated Triton code (with ModelNew class)
            problem_path: Path to problem file (with get_inputs)
            warmup: Number of warmup iterations
            runs: Number of profiled iterations
            
        Returns:
            Dict with profiling results
        """
        result = {
            "success": False,
            "error": None,
            "kernels": [],  # List of kernel metrics
            "summary": {},
            "optimization_hints": []
        }
        
        try:
            # Create profiling script
            profile_script = self._create_profile_script(code_path, problem_path, warmup, runs)
            
            # Run rocprofv3 with PMC counters
            db_path = self._run_rocprof(profile_script)
            
            if db_path and os.path.exists(db_path):
                # Parse results
                result["kernels"] = self._parse_results(db_path)
                result["summary"] = self._create_summary(result["kernels"])
                result["optimization_hints"] = self._generate_hints(result["kernels"], result["summary"])
                result["success"] = True
                
                # Cleanup
                os.unlink(db_path)
            else:
                result["error"] = "rocprofv3 failed to generate output database"
                
            os.unlink(profile_script)
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    def _create_profile_script(self, code_path: str, problem_path: str, 
                                warmup: int, runs: int) -> str:
        """Create Python script for profiling."""
        script = f'''
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

# Warmup
with torch.no_grad():
    for _ in range({warmup}):
        _ = model(*inputs)
    torch.cuda.synchronize()
    
    # Profiled runs
    for _ in range({runs}):
        _ = model(*inputs)
    torch.cuda.synchronize()

print("Profiling complete")
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            return f.name
    
    def _run_rocprof(self, script_path: str) -> Optional[str]:
        """Run rocprofv3 with PMC counters."""
        db_path = tempfile.mktemp(suffix='_results.db')
        output_prefix = db_path.replace('_results.db', '')
        
        # Build counter string
        counter_str = ",".join(self.counters)
        
        cmd = [
            'rocprofv3',
            '--pmc', counter_str,
            '--kernel-trace',
            '-o', output_prefix,
            '--',
            'python3', script_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=os.environ
            )
            
            if os.path.exists(db_path):
                return db_path
            else:
                print(f"rocprofv3 stderr: {result.stderr[:500]}")
                return None
                
        except subprocess.TimeoutExpired:
            print("rocprofv3 timed out")
            return None
        except Exception as e:
            print(f"rocprofv3 error: {e}")
            return None
    
    def _parse_results(self, db_path: str) -> List[Dict]:
        """Parse rocprofv3 SQLite database."""
        kernels = []
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get unique kernels with their metrics
            cursor.execute("""
                SELECT 
                    kernel_name,
                    counter_name,
                    SUM(value) as total_value,
                    AVG(value) as avg_value,
                    MAX(vgpr_count) as vgpr_count,
                    MAX(sgpr_count) as sgpr_count,
                    MAX(lds_block_size) as lds_size,
                    MAX(workgroup_size) as workgroup_size,
                    MAX(grid_size) as grid_size,
                    AVG(duration) as avg_duration_ns,
                    COUNT(*) as dispatch_count
                FROM counters_collection
                WHERE kernel_name NOT LIKE '%at::native%'  -- Skip PyTorch kernels
                  AND kernel_name NOT LIKE '%distribution%'
                  AND kernel_name NOT LIKE '%elementwise_kernel%'
                GROUP BY kernel_name, counter_name
            """)
            
            rows = cursor.fetchall()
            
            # Group by kernel name
            kernel_data = {}
            for row in rows:
                kernel_name = row[0]
                counter_name = row[1]
                
                if kernel_name not in kernel_data:
                    kernel_data[kernel_name] = {
                        "name": kernel_name,
                        "vgpr_count": row[4] or 0,
                        "sgpr_count": row[5] or 0,
                        "lds_size_bytes": row[6] or 0,
                        "workgroup_size": row[7] or 0,
                        "grid_size": row[8] or 0,
                        "avg_duration_ns": row[9] or 0,
                        "dispatch_count": row[10] or 0,
                        "counters": {}
                    }
                
                kernel_data[kernel_name]["counters"][counter_name] = {
                    "total": row[2] or 0,
                    "avg": row[3] or 0
                }
            
            kernels = list(kernel_data.values())
            conn.close()
            
        except Exception as e:
            print(f"Database parse error: {e}")
            
        return kernels
    
    def _create_summary(self, kernels: List[Dict]) -> Dict:
        """Create summary from kernel metrics."""
        if not kernels:
            return {}
        
        # Find the main kernel (typically the one with most dispatches or longest duration)
        main_kernel = max(kernels, key=lambda k: k["dispatch_count"] * k["avg_duration_ns"])
        
        counters = main_kernel.get("counters", {})
        
        # Calculate derived metrics
        valu_insts = counters.get("SQ_INSTS_VALU", {}).get("total", 0)
        salu_insts = counters.get("SQ_INSTS_SALU", {}).get("total", 0)
        mfma_insts = counters.get("SQ_INSTS_MFMA", {}).get("total", 0)
        waves = counters.get("SQ_WAVES", {}).get("total", 0)
        tcc_hit = counters.get("TCC_HIT", {}).get("total", 0)
        tcc_miss = counters.get("TCC_MISS", {}).get("total", 0)
        lds_conflicts = counters.get("SQ_LDS_BANK_CONFLICT", {}).get("total", 0)
        
        # Compute ratios
        total_insts = valu_insts + salu_insts + mfma_insts
        tcc_total = tcc_hit + tcc_miss
        
        return {
            "kernel_name": main_kernel["name"][:60],
            "vgpr_count": main_kernel["vgpr_count"],
            "sgpr_count": main_kernel["sgpr_count"],
            "lds_size_kb": main_kernel["lds_size_bytes"] / 1024,
            "workgroup_size": main_kernel["workgroup_size"],
            "grid_size": main_kernel["grid_size"],
            "avg_duration_us": main_kernel["avg_duration_ns"] / 1000,
            "total_waves": waves,
            "total_valu_insts": valu_insts,
            "total_salu_insts": salu_insts,
            "total_mfma_insts": mfma_insts,
            "valu_salu_ratio": valu_insts / salu_insts if salu_insts > 0 else float('inf'),
            "l2_hit_rate": tcc_hit / tcc_total * 100 if tcc_total > 0 else 0,
            "lds_bank_conflicts": lds_conflicts,
            "insts_per_wave": total_insts / waves if waves > 0 else 0,
        }
    
    def _generate_hints(self, kernels: List[Dict], summary: Dict) -> List[str]:
        """Generate optimization hints based on metrics."""
        hints = []
        
        if not summary:
            return ["No kernel metrics collected"]
        
        # 1. Check VGPR usage (occupancy impact)
        vgpr = summary.get("vgpr_count", 0)
        if vgpr > 128:
            hints.append(f"HIGH VGPR ({vgpr}): Reduce register pressure for better occupancy")
        elif vgpr > 64:
            hints.append(f"MODERATE VGPR ({vgpr}): Good occupancy potential")
        
        # 2. Check LDS usage
        lds_kb = summary.get("lds_size_kb", 0)
        if lds_kb > 48:
            hints.append(f"HIGH LDS ({lds_kb:.1f}KB): May limit CUs per workgroup")
        
        # 3. Check L2 cache hit rate
        l2_hit = summary.get("l2_hit_rate", 0)
        if l2_hit < 50:
            hints.append(f"LOW L2 HIT RATE ({l2_hit:.1f}%): Consider tile grouping for better cache reuse")
        elif l2_hit > 90:
            hints.append(f"GOOD L2 HIT RATE ({l2_hit:.1f}%): Memory access well optimized")
        
        # 4. Check LDS bank conflicts
        conflicts = summary.get("lds_bank_conflicts", 0)
        waves = summary.get("total_waves", 0)
        if waves > 0 and conflicts / waves > 1:
            hints.append(f"LDS BANK CONFLICTS ({conflicts/waves:.1f} per wave): Optimize LDS access patterns")
        
        # 5. Check instruction mix
        valu = summary.get("total_valu_insts", 0)
        salu = summary.get("total_salu_insts", 0)
        mfma = summary.get("total_mfma_insts", 0)
        
        if mfma == 0 and "gemm" in summary.get("kernel_name", "").lower():
            hints.append("NO MFMA: GEMM kernel should use matrix instructions for MI350")
        
        if salu > valu * 2:
            hints.append(f"HIGH SALU/VALU RATIO ({salu/valu:.1f}x): Too much scalar work, vectorize")
        
        # 6. Check workgroup size
        wg_size = summary.get("workgroup_size", 0)
        if wg_size < 256:
            hints.append(f"SMALL WORKGROUP ({wg_size}): Consider larger blocks for better occupancy")
        
        # 7. Check grid size vs CUs
        grid_size = summary.get("grid_size", 0)
        if grid_size < 256:
            hints.append(f"SMALL GRID ({grid_size}): May not fully utilize 256 CUs on MI350")
        
        return hints


def profile_and_report(code_path: str, problem_path: str) -> Dict:
    """Profile kernel and return formatted report."""
    profiler = RocprofProfiler()
    result = profiler.profile_kernel(code_path, problem_path)
    
    if result["success"]:
        summary = result["summary"]
        
        report = {
            "status": "success",
            "kernel_name": summary.get("kernel_name", "unknown"),
            "resource_usage": {
                "vgpr": summary.get("vgpr_count", 0),
                "sgpr": summary.get("sgpr_count", 0),
                "lds_kb": summary.get("lds_size_kb", 0),
            },
            "execution": {
                "workgroup_size": summary.get("workgroup_size", 0),
                "grid_size": summary.get("grid_size", 0),
                "avg_duration_us": summary.get("avg_duration_us", 0),
                "total_waves": summary.get("total_waves", 0),
            },
            "instruction_mix": {
                "valu": summary.get("total_valu_insts", 0),
                "salu": summary.get("total_salu_insts", 0),
                "mfma": summary.get("total_mfma_insts", 0),
                "insts_per_wave": summary.get("insts_per_wave", 0),
            },
            "memory": {
                "l2_hit_rate_pct": summary.get("l2_hit_rate", 0),
                "lds_bank_conflicts": summary.get("lds_bank_conflicts", 0),
            },
            "optimization_hints": result["optimization_hints"],
        }
    else:
        report = {
            "status": "error",
            "error": result["error"],
        }
    
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True, help="Path to Triton code")
    parser.add_argument("--problem", required=True, help="Path to problem file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    result = profile_and_report(args.code, args.problem)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["status"] == "success":
            print("=" * 60)
            print(f"KERNEL: {result['kernel_name']}")
            print("=" * 60)
            print(f"\nResource Usage:")
            print(f"  VGPR: {result['resource_usage']['vgpr']}")
            print(f"  SGPR: {result['resource_usage']['sgpr']}")
            print(f"  LDS:  {result['resource_usage']['lds_kb']:.1f} KB")
            
            print(f"\nExecution:")
            print(f"  Workgroup: {result['execution']['workgroup_size']} threads")
            print(f"  Grid:      {result['execution']['grid_size']} blocks")
            print(f"  Duration:  {result['execution']['avg_duration_us']:.2f} us")
            print(f"  Waves:     {result['execution']['total_waves']}")
            
            print(f"\nInstruction Mix:")
            print(f"  VALU: {result['instruction_mix']['valu']:,}")
            print(f"  SALU: {result['instruction_mix']['salu']:,}")
            print(f"  MFMA: {result['instruction_mix']['mfma']:,}")
            
            print(f"\nMemory:")
            print(f"  L2 Hit Rate: {result['memory']['l2_hit_rate_pct']:.1f}%")
            print(f"  LDS Conflicts: {result['memory']['lds_bank_conflicts']}")
            
            print(f"\nOptimization Hints:")
            for hint in result['optimization_hints']:
                print(f"  - {hint}")
        else:
            print(f"Error: {result['error']}")

