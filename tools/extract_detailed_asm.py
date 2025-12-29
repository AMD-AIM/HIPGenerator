#!/usr/bin/env python3
"""
深度对比 rocBLAS 和 Triton ASM:
1. 寄存器使用 (VGPR/SGPR/AGPR)
2. Pipeline stages
3. 循环结构
4. 同步原语
"""

import torch
import os
import subprocess
import tempfile
import re
from pathlib import Path

os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'

def profile_with_att(script_content: str, output_dir: str, name: str):
    """使用 rocprofv3 --att 提取详细汇编"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "rocprofv3",
        "--att=true",
        "--att-library-path", "/opt/rocm/lib",
        "-d", output_dir,
        "--", "python3", script_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(f"{name}: {result.returncode}")
        
        # 查找 code.json
        for ui_dir in Path(output_dir).glob("ui_*"):
            code_json = ui_dir / "code.json"
            if code_json.exists():
                return code_json
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        os.unlink(script_path)


def analyze_asm_file(asm_file: str):
    """分析 ASM 文件的关键指标"""
    with open(asm_file) as f:
        asm = f.read()
    
    analysis = {
        # 寄存器
        'vgpr_count': 0,
        'sgpr_count': 0,
        'agpr_count': 0,  # Accumulator VGPR (用于 MFMA)
        
        # MFMA 指令
        'mfma_total': 0,
        'mfma_f16_16x16': 0,
        'mfma_f16_32x32': 0,
        'mfma_bf16_16x16': 0,
        
        # 内存操作
        'global_load_dwordx4': 0,
        'global_load_dwordx8': 0,
        'ds_read_b128': 0,
        'ds_write_b128': 0,
        
        # 同步
        's_waitcnt': 0,
        's_waitcnt_lgkmcnt_0': 0,
        's_waitcnt_lgkmcnt_N': 0,  # N > 0
        's_waitcnt_vmcnt_0': 0,
        's_setprio': 0,
        's_barrier': 0,
        
        # Pipeline
        'main_loop_unroll': 0,
        'prefetch_count': 0,
    }
    
    lines = asm.split('\n')
    in_main_loop = False
    loop_insts = 0
    
    for line in lines:
        line = line.strip()
        
        # 寄存器使用
        if '.vgpr_count:' in line:
            m = re.search(r'(\d+)', line)
            if m: analysis['vgpr_count'] = int(m.group(1))
        if '.sgpr_count:' in line:
            m = re.search(r'(\d+)', line)
            if m: analysis['sgpr_count'] = int(m.group(1))
        if '.accum_vgpr_count:' in line or 'agpr' in line.lower():
            m = re.search(r'(\d+)', line)
            if m: analysis['agpr_count'] = int(m.group(1))
        
        # MFMA
        if 'v_mfma_f16_16x16' in line:
            analysis['mfma_f16_16x16'] += 1
            analysis['mfma_total'] += 1
        elif 'v_mfma_f16_32x32' in line:
            analysis['mfma_f16_32x32'] += 1
            analysis['mfma_total'] += 1
        elif 'v_mfma_bf16_16x16' in line:
            analysis['mfma_bf16_16x16'] += 1
            analysis['mfma_total'] += 1
        elif 'v_mfma' in line:
            analysis['mfma_total'] += 1
        
        # 内存
        if 'global_load_dwordx4' in line or 'buffer_load_dwordx4' in line:
            analysis['global_load_dwordx4'] += 1
        if 'global_load_dwordx8' in line or 'buffer_load_dwordx8' in line:
            analysis['global_load_dwordx8'] += 1
        if 'ds_read_b128' in line or 'ds_read2_b64' in line:
            analysis['ds_read_b128'] += 1
        if 'ds_write_b128' in line or 'ds_write2_b64' in line:
            analysis['ds_write_b128'] += 1
        
        # 同步
        if 's_waitcnt' in line:
            analysis['s_waitcnt'] += 1
            if 'lgkmcnt(0)' in line:
                analysis['s_waitcnt_lgkmcnt_0'] += 1
            elif 'lgkmcnt' in line:
                analysis['s_waitcnt_lgkmcnt_N'] += 1
            if 'vmcnt(0)' in line:
                analysis['s_waitcnt_vmcnt_0'] += 1
        if 's_setprio' in line:
            analysis['s_setprio'] += 1
        if 's_barrier' in line:
            analysis['s_barrier'] += 1
        
        # 主循环检测
        if 's_cbranch_scc' in line or 's_branch' in line:
            if loop_insts > 100:  # 大循环
                analysis['main_loop_unroll'] = loop_insts
            loop_insts = 0
        else:
            loop_insts += 1
    
    return analysis


def print_comparison(rocblas: dict, triton: dict):
    """打印对比结果"""
    print("\n" + "="*80)
    print("DETAILED ASM COMPARISON: rocBLAS vs Triton")
    print("="*80)
    
    categories = [
        ("REGISTERS", ['vgpr_count', 'sgpr_count', 'agpr_count']),
        ("MFMA", ['mfma_total', 'mfma_f16_16x16', 'mfma_f16_32x32']),
        ("MEMORY", ['global_load_dwordx4', 'global_load_dwordx8', 'ds_read_b128', 'ds_write_b128']),
        ("SYNC", ['s_waitcnt', 's_waitcnt_lgkmcnt_0', 's_waitcnt_lgkmcnt_N', 's_waitcnt_vmcnt_0', 's_setprio', 's_barrier']),
    ]
    
    for cat_name, metrics in categories:
        print(f"\n--- {cat_name} ---")
        print(f"{'Metric':<30} {'rocBLAS':>12} {'Triton':>12} {'Diff':>12}")
        print("-"*66)
        for m in metrics:
            r = rocblas.get(m, 0)
            t = triton.get(m, 0)
            diff = t - r
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            
            # 标记问题
            flag = ""
            if m == 'vgpr_count' and t < r * 0.5:
                flag = " ⚠️ Triton用更少寄存器"
            if m == 's_setprio' and t == 0 and r > 0:
                flag = " ⚠️ Triton缺少优先级控制"
            if m == 's_waitcnt_lgkmcnt_0' and t > r:
                flag = " ⚠️ Triton更多完全等待"
            
            print(f"{m:<30} {r:>12} {t:>12} {diff_str:>12}{flag}")


if __name__ == "__main__":
    # 生成 Triton ASM
    print("[1] Generating Triton ASM...")
    os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_deep_analysis'
    os.system('rm -rf /tmp/triton_deep_analysis')
    
    import triton
    import triton.language as tl
    
    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                      stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      GROUP_M: tl.constexpr):
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
    
    M, N, K = 4096, 4096, 4096
    a = torch.randn(M, K, dtype=torch.float16, device='cuda')
    b = torch.randn(K, N, dtype=torch.float16, device='cuda')
    c = torch.empty(M, N, dtype=torch.float16, device='cuda')
    
    grid = (triton.cdiv(M, 256) * triton.cdiv(N, 256),)
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        BLOCK_M=256, BLOCK_N=256, BLOCK_K=32, GROUP_M=8,
                        num_stages=3, num_warps=8, matrix_instr_nonkdim=16)
    
    # 找到 ASM 文件
    import glob
    triton_asm = None
    for f in glob.glob('/tmp/triton_deep_analysis/**/*.amdgcn', recursive=True):
        triton_asm = f
        print(f"  Found: {f}")
        break
    
    if triton_asm:
        triton_analysis = analyze_asm_file(triton_asm)
        print(f"  VGPR: {triton_analysis['vgpr_count']}, MFMA: {triton_analysis['mfma_total']}")
        
        # 保存 Triton ASM
        import shutil
        shutil.copy(triton_asm, '/root/HipGenerator/deep_analysis/triton_matmul.s')
        print("  Saved to triton_matmul.s")
    else:
        print("  ERROR: No ASM found")
        triton_analysis = {}
    
    print(triton_analysis)
