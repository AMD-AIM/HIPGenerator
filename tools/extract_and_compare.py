#!/usr/bin/env python3
"""
Extract and compare rocBLAS vs Triton assembly and performance.
"""

import torch
import triton
import triton.language as tl
import subprocess
import tempfile
import os
import re
from pathlib import Path

os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'

NUM_XCDS = 32


# =============== Triton Kernel (from golden solutions) ===============
@triton.jit
def matmul_kernel_aligned(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_XCDS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pids = num_pid_m * num_pid_n
    
    pids_per_xcd = (num_pids + NUM_XCDS - 1) // NUM_XCDS
    xcd_id = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if local_pid < pids_per_xcd:
        remapped_pid = xcd_id * pids_per_xcd + local_pid
        if remapped_pid < num_pids:
            pid = remapped_pid
    
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


def get_triton_asm(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps):
    """Get Triton-generated ASM for given config."""
    a = torch.randn(M, K, dtype=torch.float16, device='cuda')
    b = torch.randn(K, N, dtype=torch.float16, device='cuda')
    c = torch.empty(M, N, dtype=torch.float16, device='cuda')
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    # Compile and get ASM
    compiled = matmul_kernel_aligned[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=8, NUM_XCDS=NUM_XCDS,
        num_stages=num_stages, num_warps=num_warps, matrix_instr_nonkdim=16,
    )
    
    # Get ASM from cache
    key = matmul_kernel_aligned.cache[next(iter(matmul_kernel_aligned.cache))]
    if hasattr(key, 'asm'):
        return key.asm.get('amdgcn', '')
    return ""


def extract_rocblas_asm(M, N, K):
    """Extract rocBLAS kernel ASM using rocgdb/objdump."""
    script = f'''
import torch
a = torch.randn({M}, {K}, dtype=torch.float16, device='cuda')
b = torch.randn({K}, {N}, dtype=torch.float16, device='cuda')
for _ in range(3):
    c = torch.matmul(a, b)
torch.cuda.synchronize()
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name
    
    # Use rocprofv3 with ATT to capture assembly
    output_dir = tempfile.mkdtemp()
    cmd = [
        "rocprofv3",
        "--att=true",
        "--att-library-path", "/opt/rocm/lib",
        "-d", output_dir,
        "--", "python3", script_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Find code.json files
        for ui_dir in Path(output_dir).glob("ui_*"):
            code_json = ui_dir / "code.json"
            if code_json.exists():
                with open(code_json) as f:
                    content = f.read()
                    # Check if it's rocBLAS kernel
                    if 'blas' in content.lower() or 'gemm' in content.lower():
                        # Extract ASM from JSON
                        try:
                            data = __import__('json').loads(content)
                            if 'asm' in data:
                                return data['asm']
                        except:
                            pass
        return ""
    except Exception as e:
        print(f"Error extracting rocBLAS ASM: {e}")
        return ""
    finally:
        os.unlink(script_path)


def analyze_asm(asm_code: str, name: str = ""):
    """Analyze ASM code for key metrics."""
    if not asm_code:
        return {}
    
    analysis = {
        'name': name,
        'mfma_count': 0,
        'mfma_types': set(),
        'global_load': 0,
        'global_store': 0,
        'lds_read': 0,
        'lds_write': 0,
        's_waitcnt_total': 0,
        's_waitcnt_lgkm0': 0,
        's_waitcnt_vmcnt0': 0,
        's_setprio': 0,
        's_barrier': 0,
        'v_mov': 0,
        's_mov': 0,
        'total_inst': 0,
    }
    
    for line in asm_code.split('\n'):
        line = line.strip()
        if not line or line.startswith(';') or line.startswith('//'):
            continue
        
        analysis['total_inst'] += 1
        
        # MFMA
        if 'v_mfma' in line:
            analysis['mfma_count'] += 1
            match = re.search(r'v_mfma_\w+', line)
            if match:
                analysis['mfma_types'].add(match.group())
        
        # Memory
        if 'global_load' in line or 'buffer_load' in line:
            analysis['global_load'] += 1
        if 'global_store' in line or 'buffer_store' in line:
            analysis['global_store'] += 1
        if 'ds_read' in line:
            analysis['lds_read'] += 1
        if 'ds_write' in line:
            analysis['lds_write'] += 1
        
        # Sync
        if 's_waitcnt' in line:
            analysis['s_waitcnt_total'] += 1
            if 'lgkmcnt(0)' in line:
                analysis['s_waitcnt_lgkm0'] += 1
            if 'vmcnt(0)' in line:
                analysis['s_waitcnt_vmcnt0'] += 1
        if 's_setprio' in line:
            analysis['s_setprio'] += 1
        if 's_barrier' in line:
            analysis['s_barrier'] += 1
        
        # MOV
        if line.startswith('v_mov'):
            analysis['v_mov'] += 1
        if line.startswith('s_mov'):
            analysis['s_mov'] += 1
    
    analysis['mfma_types'] = list(analysis['mfma_types'])
    return analysis


def run_perf_counters(M, N, K, output_dir):
    """Run rocprofv3 with performance counters."""
    script = f'''
import torch
import os
os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'

# rocBLAS baseline
a = torch.randn({M}, {K}, dtype=torch.float16, device='cuda')
b = torch.randn({K}, {N}, dtype=torch.float16, device='cuda')
for _ in range(5):
    c_ref = torch.matmul(a, b)
torch.cuda.synchronize()

# Triton kernel
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))

c_triton = torch.empty({M}, {N}, dtype=torch.float16, device='cuda')
grid = (triton.cdiv({M}, 256) * triton.cdiv({N}, 256),)
for _ in range(5):
    matmul_kernel[grid](
        a, b, c_triton, {M}, {N}, {K},
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_triton.stride(0), c_triton.stride(1),
        BLOCK_M=256, BLOCK_N=256, BLOCK_K=32, GROUP_M=8,
        num_stages=3, num_warps=8, matrix_instr_nonkdim=16,
    )
torch.cuda.synchronize()
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Key counters
    counters = "SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_VMEM,SQ_INSTS_LDS,SQ_INSTS_MFMA,TCP_TCC_READ_REQ_sum,TCC_HIT_sum,TCC_MISS_sum,SQ_LDS_BANK_CONFLICT"
    
    cmd = [
        "rocprofv3",
        f"--pmc={counters}",
        "-d", output_dir,
        "--", "python3", script_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr[:500])
        
        # Parse CSV results
        return parse_counter_csv(output_dir)
    finally:
        os.unlink(script_path)


def parse_counter_csv(output_dir):
    """Parse rocprofv3 counter CSV output."""
    results = {}
    
    for csv_file in Path(output_dir).glob("**/*.csv"):
        with open(csv_file) as f:
            lines = f.readlines()
            if len(lines) > 1:
                headers = [h.strip() for h in lines[0].split(',')]
                for line in lines[1:]:
                    values = [v.strip() for v in line.split(',')]
                    if len(values) >= len(headers):
                        name = values[0] if values else "unknown"
                        results[name] = dict(zip(headers, values))
    
    return results


def main():
    print("="*80)
    print("COMPREHENSIVE KERNEL ANALYSIS: rocBLAS vs Triton")
    print("Target: MI350 (gfx950, 32 XCDs)")
    print("="*80)
    
    # Test case: 4096x4096 square matrix
    M, N, K = 4096, 4096, 4096
    
    print(f"\n>>> Problem: {M}x{K} @ {K}x{N}")
    
    # 1. Get Triton ASM
    print("\n[1] Extracting Triton ASM...")
    try:
        triton_asm = get_triton_asm(M, N, K, 256, 256, 32, 3, 8)
        triton_analysis = analyze_asm(triton_asm, "Triton")
        
        # Save ASM
        with open("/root/HipGenerator/profiling_analysis/triton_4096.s", 'w') as f:
            f.write(triton_asm)
        print(f"   Saved to triton_4096.s ({len(triton_asm)} bytes)")
        
        print(f"\n   Triton Analysis:")
        for k, v in triton_analysis.items():
            if v and k != 'name':
                print(f"     {k}: {v}")
    except Exception as e:
        print(f"   Error: {e}")
        triton_analysis = {}
    
    # 2. Run performance counters
    print("\n[2] Running rocprofv3 performance counters...")
    output_dir = "/root/HipGenerator/profiling_analysis/perf_counters"
    try:
        counter_results = run_perf_counters(M, N, K, output_dir)
        
        print(f"\n   Counter Results:")
        for kernel, counters in counter_results.items():
            kernel_short = kernel[:60] + "..." if len(kernel) > 60 else kernel
            print(f"\n   Kernel: {kernel_short}")
            for k, v in counters.items():
                if k not in ['Name', 'Kind', 'Agent_Id', 'Queue_Id', 'Process_Id']:
                    try:
                        val = float(v)
                        if val > 0:
                            print(f"     {k}: {val:,.0f}")
                    except:
                        pass
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. Optimization recommendations
    print("\n[3] OPTIMIZATION RECOMMENDATIONS")
    print("-"*60)
    
    if triton_analysis:
        lgkm0_ratio = triton_analysis.get('s_waitcnt_lgkm0', 0) / max(triton_analysis.get('s_waitcnt_total', 1), 1)
        
        if lgkm0_ratio > 0.5:
            print("⚠️  High lgkmcnt(0) ratio ({:.1%}) - Consider fine-grained LDS barriers".format(lgkm0_ratio))
        
        if triton_analysis.get('s_setprio', 0) == 0:
            print("⚠️  No s_setprio instructions - Enable pingpong scheduling")
        
        mfma_ratio = triton_analysis.get('mfma_count', 0) / max(triton_analysis.get('total_inst', 1), 1)
        if mfma_ratio < 0.3:
            print(f"⚠️  Low MFMA density ({mfma_ratio:.1%}) - Consider larger tiles or better pipelining")
        
        print("\n✓ Recommendations applied in golden solutions:")
        print("  - TRITON_HIP_USE_BLOCK_PINGPONG=1 (enables s_setprio)")
        print("  - Modified Triton lgkmcnt from 0 to 4")
        print("  - XCD swizzle for 32 XCDs")
        print("  - 256x256x32 tiles with 3 stages")


if __name__ == "__main__":
    main()


