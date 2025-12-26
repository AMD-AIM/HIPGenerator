#!/usr/bin/env python3
"""
Run all 10 golden solution benchmarks and generate report.
"""

import subprocess
import os
import sys
import time

os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'

GOLDEN_SOLUTIONS = [
    ("01_square_gemm.py", "Square GEMM 4096x4096"),
    ("02_batched_gemm.py", "Batched GEMM 128x512x1024x2048"),
    ("03_transposed_A.py", "Transposed A 2048x8192x4096"),
    ("04_gemm_bias_relu.py", "GEMM + Bias + ReLU"),
    ("05_gemm_divide_gelu.py", "GEMM + Divide + GELU"),
    ("06_tall_skinny.py", "Tall-skinny 16384x16x1024"),
    ("07_gemm_swish_scaling.py", "GEMM + Swish + Scaling"),
    ("08_rectangular_gemm.py", "Rectangular 1024x4096x2048"),
    ("09_gemm_sigmoid_sum.py", "GEMM + Sigmoid + Sum"),
    ("10_gemm_gelu_softmax.py", "GEMM + GELU + Softmax"),
]


def run_benchmark(script):
    """Run a single benchmark and parse results."""
    try:
        result = subprocess.run(
            ["python3", script],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        output = result.stdout + result.stderr
        
        # Parse speedup
        for line in output.split('\n'):
            if 'Speedup:' in line:
                speedup = float(line.split(':')[1].strip().replace('x', ''))
                return speedup, output
        
        return None, output
    except Exception as e:
        return None, str(e)


def main():
    print("=" * 80)
    print("TRITON GOLDEN SOLUTIONS BENCHMARK")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Clear cache
    os.system("rm -rf ~/.triton/cache")
    
    results = []
    
    for script, name in GOLDEN_SOLUTIONS:
        print(f"Running {script}...", end=" ", flush=True)
        speedup, _ = run_benchmark(script)
        
        if speedup:
            marker = "⭐⭐" if speedup >= 1.1 else ("⭐" if speedup >= 1.0 else ("✓" if speedup >= 0.9 else ("~" if speedup >= 0.8 else "⚠")))
            print(f"{speedup:.3f}x {marker}")
            results.append((script, name, speedup, marker))
        else:
            print("FAILED")
            results.append((script, name, None, "❌"))
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'#':<3} {'Problem':<35} {'Speedup':<10} {'Status':<6} {'Gap to 1.1x':<10}")
    print("-" * 70)
    
    total_speedup = 0
    count = 0
    exceeds_11 = 0
    exceeds_10 = 0
    exceeds_09 = 0
    exceeds_08 = 0
    
    for i, (script, name, speedup, marker) in enumerate(results, 1):
        if speedup:
            gap = (1.1 - speedup) * 100
            gap_str = f"{gap:+.1f}%" if gap > 0 else f"{gap:.1f}%"
            print(f"{i:02d}  {name:<35} {speedup:.3f}x     {marker:<6} {gap_str}")
            total_speedup += speedup
            count += 1
            if speedup >= 1.1:
                exceeds_11 += 1
            if speedup >= 1.0:
                exceeds_10 += 1
            if speedup >= 0.9:
                exceeds_09 += 1
            if speedup >= 0.8:
                exceeds_08 += 1
        else:
            print(f"{i:02d}  {name:<35} {'FAILED':<10} {marker}")
    
    print("-" * 70)
    print()
    print(f"Average speedup: {total_speedup/count:.3f}x" if count > 0 else "No valid results")
    print()
    print("Distribution:")
    print(f"  >= 1.1x (target): {exceeds_11}/{count}")
    print(f"  >= 1.0x:          {exceeds_10}/{count}")
    print(f"  >= 0.9x:          {exceeds_09}/{count}")
    print(f"  >= 0.8x:          {exceeds_08}/{count}")


if __name__ == "__main__":
    main()

