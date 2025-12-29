# Reverse Engineering & Profiling Tools

Utility scripts for analyzing GPU kernel performance and assembly.

## Tools

### extract_triton_asm.py
Extract assembly code from compiled Triton kernels.
```bash
python3 extract_triton_asm.py
```

### analyze_pingpong_asm.py
Analyze block pingpong scheduling patterns in assembly (s_setprio, ds_read/write, s_waitcnt).
```bash
python3 analyze_pingpong_asm.py <asm_file>
```

### analyze_kernel.py
Profile kernel using rocprofv3 and collect PMC counters.
```bash
python3 analyze_kernel.py --kernel <kernel_file.py>
```

### extract_and_compare.py
Extract and compare assembly between rocBLAS and Triton kernels.
```bash
python3 extract_and_compare.py
```

### extract_detailed_asm.py
Extract detailed assembly from rocBLAS/hipBLASLt library files.
```bash
python3 extract_detailed_asm.py
```

## Typical Workflow

1. **Profile baseline**: Use `analyze_kernel.py` to get PMC counters
2. **Extract ASM**: Use `extract_triton_asm.py` or `extract_detailed_asm.py`
3. **Compare**: Use `extract_and_compare.py` to identify differences
4. **Analyze patterns**: Use `analyze_pingpong_asm.py` for scheduling analysis










