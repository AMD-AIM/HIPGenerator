#!/usr/bin/env python3
"""
Comprehensive kernel analysis tool:
1. Extract and compare ASM between rocBLAS and Triton
2. Profile with rocprofv3 performance counters
3. Identify optimization opportunities
"""

import torch
import triton
import triton.language as tl
import subprocess
import tempfile
import os
import re
import json
from pathlib import Path

os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'


def extract_triton_asm(kernel_fn, *args, **kwargs):
    """Extract Triton-generated AMDGCN assembly."""
    # Compile kernel and get assembly
    grid = kwargs.pop('grid', (1,))
    pgm = kernel_fn[grid](*args, **kwargs)
    
    # Get the compiled binary info
    if hasattr(pgm, 'asm'):
        return pgm.asm.get('amdgcn', '')
    return ""


def analyze_asm(asm_code: str):
    """Analyze assembly code for key patterns."""
    analysis = {
        'mfma_count': 0,
        'mfma_type': [],
        'global_load': 0,
        'global_store': 0,
        'lds_read': 0,
        'lds_write': 0,
        's_waitcnt': [],
        's_setprio': 0,
        's_barrier': 0,
        'total_instructions': 0,
        'vgpr_used': 0,
        'sgpr_used': 0,
    }
    
    lines = asm_code.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith(';') or line.startswith('//'):
            continue
        
        analysis['total_instructions'] += 1
        
        # MFMA instructions
        if 'v_mfma' in line:
            analysis['mfma_count'] += 1
            match = re.search(r'v_mfma_\w+', line)
            if match:
                mfma_type = match.group()
                if mfma_type not in analysis['mfma_type']:
                    analysis['mfma_type'].append(mfma_type)
        
        # Memory operations
        if 'global_load' in line or 'buffer_load' in line:
            analysis['global_load'] += 1
        if 'global_store' in line or 'buffer_store' in line:
            analysis['global_store'] += 1
        if 'ds_read' in line:
            analysis['lds_read'] += 1
        if 'ds_write' in line:
            analysis['lds_write'] += 1
        
        # Synchronization
        if 's_waitcnt' in line:
            analysis['s_waitcnt'].append(line)
        if 's_setprio' in line:
            analysis['s_setprio'] += 1
        if 's_barrier' in line:
            analysis['s_barrier'] += 1
        
        # Register usage
        if '.vgpr_count:' in line:
            match = re.search(r'(\d+)', line)
            if match:
                analysis['vgpr_used'] = int(match.group(1))
        if '.sgpr_count:' in line:
            match = re.search(r'(\d+)', line)
            if match:
                analysis['sgpr_used'] = int(match.group(1))
    
    return analysis


def compare_asm(rocblas_asm: str, triton_asm: str):
    """Compare rocBLAS and Triton assembly."""
    rocblas_analysis = analyze_asm(rocblas_asm)
    triton_analysis = analyze_asm(triton_asm)
    
    print("\n" + "="*70)
    print("ASM COMPARISON: rocBLAS vs Triton")
    print("="*70)
    
    metrics = [
        ('MFMA Instructions', 'mfma_count'),
        ('Global Loads', 'global_load'),
        ('Global Stores', 'global_store'),
        ('LDS Reads', 'lds_read'),
        ('LDS Writes', 'lds_write'),
        ('s_setprio', 's_setprio'),
        ('s_barrier', 's_barrier'),
        ('Total Instructions', 'total_instructions'),
        ('VGPR Used', 'vgpr_used'),
        ('SGPR Used', 'sgpr_used'),
    ]
    
    print(f"\n{'Metric':<25} {'rocBLAS':>12} {'Triton':>12} {'Diff':>12}")
    print("-"*61)
    
    for name, key in metrics:
        r_val = rocblas_analysis[key]
        t_val = triton_analysis[key]
        diff = t_val - r_val if isinstance(t_val, (int, float)) else 'N/A'
        print(f"{name:<25} {str(r_val):>12} {str(t_val):>12} {str(diff):>12}")
    
    print("\n--- MFMA Types ---")
    print(f"rocBLAS: {rocblas_analysis['mfma_type']}")
    print(f"Triton:  {triton_analysis['mfma_type']}")
    
    print("\n--- s_waitcnt Analysis ---")
    rocblas_waits = len(rocblas_analysis['s_waitcnt'])
    triton_waits = len(triton_analysis['s_waitcnt'])
    print(f"rocBLAS s_waitcnt count: {rocblas_waits}")
    print(f"Triton s_waitcnt count:  {triton_waits}")
    
    # Check for lgkmcnt(0) vs fine-grained
    rocblas_lgkm0 = sum(1 for w in rocblas_analysis['s_waitcnt'] if 'lgkmcnt(0)' in w)
    triton_lgkm0 = sum(1 for w in triton_analysis['s_waitcnt'] if 'lgkmcnt(0)' in w)
    print(f"rocBLAS lgkmcnt(0): {rocblas_lgkm0}")
    print(f"Triton lgkmcnt(0):  {triton_lgkm0}")
    
    return rocblas_analysis, triton_analysis


def run_rocprof_counters(script_path: str, output_dir: str):
    """Run rocprofv3 with performance counters."""
    counters = [
        # Compute
        "SQ_WAVES",
        "SQ_INSTS_VALU", 
        "SQ_INSTS_VMEM",
        "SQ_INSTS_LDS",
        "SQ_INSTS_SMEM",
        "SQ_INSTS_MFMA",
        # Memory
        "TCP_TCC_READ_REQ_sum",
        "TCP_TCC_WRITE_REQ_sum", 
        "TCC_HIT_sum",
        "TCC_MISS_sum",
        # LDS
        "SQ_LDS_BANK_CONFLICT",
        # Occupancy
        "SQ_ACCUM_PREV_HIRES",
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create input file for counters
    counter_str = ",".join(counters)
    
    cmd = [
        "rocprofv3",
        f"--pmc={counter_str}",
        "-d", output_dir,
        "--", "python3", script_path
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    # Parse output
    return parse_rocprof_output(output_dir)


def parse_rocprof_output(output_dir: str):
    """Parse rocprofv3 counter output."""
    results = {}
    
    # Find CSV files
    for f in Path(output_dir).glob("*.csv"):
        with open(f) as fp:
            lines = fp.readlines()
            if len(lines) > 1:
                headers = lines[0].strip().split(',')
                for line in lines[1:]:
                    values = line.strip().split(',')
                    kernel_name = values[0] if values else ""
                    if kernel_name:
                        results[kernel_name] = dict(zip(headers, values))
    
    return results


def create_profile_script(problem_name: str, golden_dir: str):
    """Create a profiling script for a specific problem."""
    script = f'''#!/usr/bin/env python3
import torch
import sys
sys.path.insert(0, "{golden_dir}")
from {problem_name} import Model, ModelNew, get_inputs, get_init_inputs

# Setup
inputs = get_inputs()
inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

init_inputs = get_init_inputs()

# Warmup and profile reference (rocBLAS)
ref_model = Model(*init_inputs).cuda().half() if init_inputs else Model().cuda()
for _ in range(3):
    ref_out = ref_model(*inputs)
torch.cuda.synchronize()

# Warmup and profile Triton
new_model = ModelNew(*init_inputs).cuda() if init_inputs else ModelNew().cuda()
for _ in range(3):
    new_out = new_model(*inputs)
torch.cuda.synchronize()

print("Profiling complete")
'''
    return script


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_kernel.py <problem_name>")
        print("Example: python analyze_kernel.py 01_square_gemm")
        sys.exit(1)
    
    problem_name = sys.argv[1]
    golden_dir = "/root/HipGenerator/golden_solutions"
    
    # Create profiling script
    script_content = create_profile_script(problem_name, golden_dir)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # Run profiling
        output_dir = f"/root/HipGenerator/profiling_analysis/results_{problem_name}"
        results = run_rocprof_counters(script_path, output_dir)
        
        if results:
            print("\n" + "="*70)
            print(f"PERFORMANCE COUNTERS: {problem_name}")
            print("="*70)
            for kernel, counters in results.items():
                print(f"\nKernel: {kernel[:50]}...")
                for k, v in counters.items():
                    if k not in ['Name', 'Kind']:
                        print(f"  {k}: {v}")
    finally:
        os.unlink(script_path)


