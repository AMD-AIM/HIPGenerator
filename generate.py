#!/usr/bin/env python3
"""
Generate HipKittens/Triton kernel code from a KernelBench problem using LLM.
Usage: python generate.py --problem <problem_path> --output <output_path> [--prompt <prompt_file>] [--backend hip|triton]
"""
import os
import sys
import json
import argparse
import re
from pathlib import Path

# Setup environment
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

from anthropic import Anthropic

# Supported backends
BACKENDS = ["hip", "triton"]


def classify_problem(problem_name: str, problem_code: str) -> str:
    """Classify problem type based on name and code."""
    name_lower = problem_name.lower()
    
    # Matrix-Vector: has N=1 or (K, 1) shape
    if "matrix_vector" in name_lower or "matvec" in name_lower or "mv_" in name_lower:
        return "matvec"
    
    # Check for batch dimension in code
    if "batch" in name_lower or "bmm" in name_lower:
        return "batched_gemm"
    
    # 2D GEMM patterns
    gemm_2d_patterns = [
        r"square.*matrix", r"standard.*matrix", r"gemm", r"mm_"
    ]
    if any(re.search(p, name_lower) for p in gemm_2d_patterns):
        return "gemm_2d"
    
    # Fallback: check if it's a matmul-like operation
    if "matrix" in name_lower and "mult" in name_lower:
        return "gemm_2d"
    
    return "unknown"


def get_template(problem_type: str) -> str:
    """Return pre-verified template for problem type."""
    script_dir = Path(__file__).parent
    
    template_map = {
        "gemm_2d": "gemm_template.py",
        "batched_gemm": "batched_gemm_template.py",
        "matvec": "matvec_template.py",
    }
    
    template_name = template_map.get(problem_type)
    if template_name:
        template_path = script_dir / "templates" / template_name
        if template_path.exists():
            return template_path.read_text()
    return None


def load_prompt(prompt_file: str, problem_name: str = None, backend: str = "hip") -> str:
    """Load system prompt from file, config, or use default.
    
    Args:
        prompt_file: Explicit prompt file path (overrides config)
        problem_name: Problem name for config-based selection
        backend: Backend type ('hip' or 'triton')
    """
    # If explicit prompt file provided, use it
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file) as f:
            return f.read()
    
    # Try to load from config based on problem name and backend
    script_dir = Path(__file__).parent
    config_path = script_dir / "prompts" / "config.json"
    
    if config_path.exists() and problem_name:
        import re
        with open(config_path) as f:
            config = json.load(f)
        
        problem_lower = problem_name.lower()
        
        # For Triton backend, use triton-specific patterns
        if backend == "triton":
            patterns_key = "triton_patterns"
            default_key = "triton_default_prompt"
        else:
            patterns_key = "patterns"
            default_key = "default_prompt"
        
        patterns = config.get(patterns_key, config.get("patterns", {}))
        
        for pattern, prompt_filename in patterns.items():
            if re.search(pattern, problem_lower, re.IGNORECASE):
                prompt_path = script_dir / "prompts" / prompt_filename
                if prompt_path.exists():
                    with open(prompt_path) as f:
                        return f.read()
        
        # Use default from config
        if backend == "triton":
            default_prompt = config.get(default_key, "triton_base.txt")
        else:
            default_prompt = config.get(default_key, "elementwise_bf16.txt")
        
        default_path = script_dir / "prompts" / default_prompt
        if default_path.exists():
            with open(default_path) as f:
                return f.read()
    
    return get_default_prompt(backend)


def get_default_prompt(backend: str = "hip") -> str:
    """Default system prompt based on backend."""
    if backend == "triton":
        return get_default_triton_prompt()
    return get_default_hip_prompt()


def get_default_triton_prompt() -> str:
    """Default Triton system prompt."""
    return '''You are an expert Triton programmer for AMD GPUs.

**CRITICAL RULES - MUST FOLLOW:**
1. ALL computation MUST be in Triton kernels using @triton.jit
2. ABSOLUTELY FORBIDDEN: torch.mm(), torch.matmul(), torch.bmm() for the main computation
3. Input dtype is torch.bfloat16 (bf16). Process bf16 directly using tl.bfloat16
4. PyTorch ONLY for: torch.empty(), .data_ptr(), .contiguous()
5. For GEMM: MUST use tl.dot() which maps to AMD MFMA instructions

**Triton GEMM template:**
```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c
```

Generate complete Python code with ModelNew class using Triton kernels.

**OUTPUT FORMAT:**
- Output ONLY the Python code inside ```python ... ``` block
- NO explanations, NO comments before or after the code block
- Start directly with ```python and end with ```
'''


def get_default_hip_prompt() -> str:
    """Default HipKittens system prompt."""
    return '''You are an expert C++/HIP programmer for AMD GPUs using HipKittens library.

**CRITICAL RULES - MUST FOLLOW:**
1. ALL computation MUST be in HIP kernels using HipKittens or native HIP
2. ABSOLUTELY FORBIDDEN in C++ code: 
   - .to(torch::kFloat32), .to(x.dtype()), ANY .to() CALL = VERY SLOW
   - torch::mm(), torch::matmul(), torch::bmm(), torch::add(), torch::relu(), torch::mv()
3. Input dtype is torch.bfloat16 (bf16). Process bf16 DIRECTLY using hip_bfloat16 type.
4. PyTorch ONLY for: torch::empty(), .data_ptr(), .contiguous()
5. For GEMM: MUST use HipKittens MFMA, NOT naive nested loops

**load_inline template (FOLLOW EXACTLY - DO NOT MODIFY):**
```python
from torch.utils.cpp_extension import load_inline

# cpp_src: ONLY function declarations, NO PYBIND11_MODULE (added automatically by load_inline)!
cpp_src = \'\'\'
#include <torch/extension.h>
torch::Tensor forward_func(torch::Tensor a);
\'\'\'

hip_src = \'\'\'
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Kernel implementation here...

torch::Tensor forward_func(torch::Tensor a) {
    // Implementation here...
}
\'\'\'

module = load_inline(
    name="module_name",
    cpp_sources=cpp_src,
    cuda_sources=hip_src,  # MUST use cuda_sources, NOT hip_sources!
    functions=["forward_func"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "-std=c++20", "-I/root/agent/HipKittens/include",
                       "-I/opt/rocm/include/hip", "-I/opt/rocm/include/rocrand",
                       "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
                       "--offload-arch=gfx950"],
    verbose=False
)
```

**CRITICAL: cpp_src must NOT contain PYBIND11_MODULE - load_inline adds it automatically!**

**PERFORMANCE CRITICAL: NEVER use .to() for type conversion - it's extremely slow!**
Process input tensors DIRECTLY in their original dtype (bfloat16/float32).
Input dtype is ALWAYS torch.bfloat16. Use hip_bfloat16 type directly.

**For SIMPLE element-wise ops (ReLU, Sigmoid, Tanh, etc.), use vectorized HIP with bf16:**
NOTE: Use `hip_bfloat16` type from `<hip/hip_bfloat16.h>`. Use float4 for 128-bit vectorized loads (8 bf16 values).
```cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Process 8 bf16 values per thread using float4 (128 bits = 8 bf16)
__global__ void relu_kernel_bf16(const float4* __restrict__ in, 
                                  float4* __restrict__ out, int64_t size4) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size4) {
        float4 data = in[idx];
        hip_bfloat16* vals = reinterpret_cast<hip_bfloat16*>(&data);
        
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float v = static_cast<float>(vals[i]);
            vals[i] = hip_bfloat16(fmaxf(v, 0.0f));  // ReLU
        }
        out[idx] = data;
    }
}

torch::Tensor relu_forward(torch::Tensor x) {
    auto x_cont = x.contiguous();
    auto output = torch::empty_like(x_cont);
    int64_t numel = x_cont.numel();
    int64_t numel8 = numel / 8;  // Number of float4 chunks (8 bf16 each)
    
    int threads = 256;
    int blocks = (numel8 + threads - 1) / threads;
    
    relu_kernel_bf16<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x_cont.data_ptr<at::BFloat16>()),
        reinterpret_cast<float4*>(output.data_ptr<at::BFloat16>()),
        numel8);
    return output;  // NO .to() conversion - already bf16!
}
```

**For LayerNorm/Softmax (each warp processes one row, row_size <= 2048), use HipKittens:**
```cpp
using input_gl = gl<bf16, -1, -1, -1, -1>;
constexpr int D = 2048;

__global__ void kernel(input_gl g_in, input_gl g_out, int N) {
    int row = blockIdx.x * 4 + kittens::warpid(); // 4 warps per block
    if (row >= N) return;
    
    rv_naive<bf16, D> x_reg;
    load(x_reg, g_in, {0, 0, row, 0});
    asm volatile("s_waitcnt vmcnt(0)");  // REQUIRED after load
    
    // Process x_reg using HipKittens ops: add, mul, sub, sum
    // Access elements: x_reg[0][i] where i < D/64
    
    store(g_out, x_reg, {0, 0, row, 0});
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "Requires bf16 input");
    auto x_cont = x.contiguous();
    auto output = torch::empty_like(x_cont);
    
    // IMPORTANT: Create gl using make_gl
    auto g_in = make_gl<input_gl>((uint64_t)x_cont.data_ptr<at::BFloat16>(), 1, 1, N, D);
    auto g_out = make_gl<input_gl>((uint64_t)output.data_ptr<at::BFloat16>(), 1, 1, N, D);
    
    kernel<<<blocks, threads>>>(g_in, g_out, N);
    return output;  // NO .to() conversion!
}
```

**For GEMM/MatMul (C = A @ B), use HipKittens MFMA - REQUIRED for matrix multiply:**
```cpp
#include "kittens.cuh"
using namespace kittens;

// Type aliases
using _gl_bf16 = gl<bf16, 1, 1, -1, -1>;  // Global layout for bf16
using _st_bf16 = st_bf<64, 64>;           // Shared tile 64x64 bf16
using _rt_bf16 = rt_bf<16, 16>;           // Register tile 16x16 bf16 for MFMA input
using _rt_fl32 = rt_fl<16, 16>;           // Register tile 16x16 float for accumulator

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;
constexpr int WARPS = 4;  // 4 warps per block

__global__ __launch_bounds__(WARPS * 64, 1)
void gemm_mfma_kernel(_gl_bf16 g_A, _gl_bf16 g_B, _gl_bf16 g_C, int M, int N, int K) {
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    int warp_id = threadIdx.x / 64;
    
    // Shared memory for tiles
    extern __shared__ char smem[];
    _st_bf16& sA = *reinterpret_cast<_st_bf16*>(smem);
    _st_bf16& sB = *reinterpret_cast<_st_bf16*>(smem + sizeof(_st_bf16));
    
    // Register accumulator (float32 for precision)
    _rt_fl32 acc;
    zero(acc);
    
    // Loop over K dimension
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load A tile [BLOCK_M, BLOCK_K] and B tile [BLOCK_K, BLOCK_N] to shared
        load(sA, g_A, {0, 0, block_m * BLOCK_M, k});
        load(sB, g_B, {0, 0, k, block_n * BLOCK_N});
        __builtin_amdgcn_s_barrier();
        
        // Register tiles for MFMA
        _rt_bf16 rA, rB;
        load(rA, sA, warp_id * 16, 0);  // Each warp loads 16 rows
        load(rB, sB, 0, warp_id * 16);  // Each warp loads 16 cols
        
        // MFMA: acc += A @ B^T (note: mma_ABt expects B transposed)
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc, rA, rB, acc);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }
    
    // Store result
    _rt_bf16 result;
    copy(result, acc);  // Convert float32 acc to bf16
    store(g_C, result, {0, 0, block_m * BLOCK_M + warp_id * 16, block_n * BLOCK_N});
}

torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16, "A must be bf16");
    TORCH_CHECK(B.scalar_type() == torch::kBFloat16, "B must be bf16");
    
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    
    _gl_bf16 g_A{(bf16*)A.data_ptr<at::BFloat16>(), 1u, 1u, (unsigned)M, (unsigned)K};
    _gl_bf16 g_B{(bf16*)B.data_ptr<at::BFloat16>(), 1u, 1u, (unsigned)K, (unsigned)N};
    _gl_bf16 g_C{(bf16*)C.data_ptr<at::BFloat16>(), 1u, 1u, (unsigned)M, (unsigned)N};
    
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(WARPS * 64);
    size_t smem = 2 * sizeof(_st_bf16);
    
    gemm_mfma_kernel<<<grid, block, smem>>>(g_A, g_B, g_C, M, N, K);
    return C;
}
```

Generate complete Python code with ModelNew class.

**OUTPUT FORMAT:**
- Output ONLY the Python code inside ```python ... ``` block
- NO explanations, NO comments before or after the code block
- Start directly with ```python and end with ```
'''


def load_problem(problem_path: str) -> dict:
    """Load KernelBench problem file."""
    with open(problem_path) as f:
        content = f.read()
    
    return {
        "path": problem_path,
        "code": content,
        "name": Path(problem_path).stem
    }


def build_user_prompt(problem: dict, backend: str = "hip") -> str:
    """Build user prompt from problem."""
    if backend == "triton":
        return f'''**PyTorch Reference Implementation:**
```python
{problem["code"]}
```

Generate complete Python code implementing ModelNew with Triton kernels.
Include: Triton kernel(s) with @triton.autotune, ModelNew class.
The ModelNew.forward() must produce same output as Model.forward().
Use tl.dot() for matrix multiplications, NOT torch.mm/matmul.
'''
    else:
        return f'''**PyTorch Reference Implementation:**
```python
{problem["code"]}
```

Generate complete Python code implementing ModelNew with HipKittens.
Include: cpp_src, hip_src, load_inline, ModelNew class.
The ModelNew.forward() must produce same output as Model.forward().
'''


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Call LLM API to generate code."""
    client = Anthropic(
        base_url="https://llm-api.amd.com/Anthropic",
        api_key="dummy",
        default_headers={
            "Ocp-Apim-Subscription-Key": os.environ.get("LLM_GATEWAY_KEY"),
        }
    )
    
    response = client.messages.create(
        model="claude-opus-4",
        max_tokens=16384,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    
    return response.content[0].text


def generate_samples(system_prompt: str, user_prompt: str, num_samples: int = 3, 
                     temperature: float = 0.7, backend: str = "hip") -> list:
    """Generate multiple code samples in parallel using threading."""
    import concurrent.futures
    
    samples = []
    
    # Strategy hints for Triton backend to increase diversity
    triton_strategy_hints = [
        "\n\n**IMPORTANT: For this sample, use Strategy 1 (Split-K) or Strategy 4 (Large 256x256 Tiles).**",
        "\n\n**IMPORTANT: For this sample, use different autotune configs with BLOCK_K=128 for AMD MFMA.**",
        "\n\n**IMPORTANT: For this sample, try unique tile sizes like 256x64 or 64x256 with high num_stages.**",
    ]
    
    def generate_one(sample_id):
        """Generate a single sample."""
        # Larger temperature variation for diversity
        temp = temperature + (sample_id - 1) * 0.15  # e.g., 0.3, 0.45, 0.6 for 3 samples
        temp = min(temp, 0.8)  # Allow higher temperature for more diversity
        
        # Add strategy hint for Triton to force different approaches
        if backend == "triton" and sample_id <= len(triton_strategy_hints):
            modified_prompt = user_prompt + triton_strategy_hints[sample_id - 1]
        else:
            modified_prompt = user_prompt
        
        try:
            response = call_llm(system_prompt, modified_prompt, temperature=temp)
            code = extract_code(response)
            return {"id": sample_id, "response": response, "code": code, "error": None}
        except Exception as e:
            return {"id": sample_id, "response": None, "code": None, "error": str(e)}
    
    # Generate samples in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_samples) as executor:
        futures = [executor.submit(generate_one, i) for i in range(1, num_samples + 1)]
        for future in concurrent.futures.as_completed(futures):
            samples.append(future.result())
    
    # Sort by sample ID
    samples.sort(key=lambda x: x["id"])
    return samples


def extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    # Try to find complete Python code block
    pattern = r'```python\s*(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Handle incomplete code block (no closing ```)
    if '```python' in response:
        start = response.find('```python') + len('```python')
        code = response[start:].strip()
        # Remove any trailing incomplete lines
        lines = code.split('\n')
        # Find the last complete line (ends with valid Python syntax)
        while lines and not lines[-1].strip():
            lines.pop()
        return '\n'.join(lines)
    
    # Try to find code starting with import
    if 'import torch' in response:
        start = response.find('import torch')
        return response[start:].strip()
    
    # If no code block, return the whole response
    return response.strip()


def load_optimize_prompt() -> str:
    """Load the Triton optimization prompt."""
    script_dir = Path(__file__).parent
    prompt_path = script_dir / "prompts" / "triton_optimize.txt"
    
    if prompt_path.exists():
        with open(prompt_path) as f:
            return f.read()
    
    # Fallback inline prompt
    return '''You are an expert Triton kernel optimizer for AMD MI350 GPUs.
Optimize the given Triton kernel for maximum performance.

Key optimizations to apply:
1. XCD Swizzle (32 XCDs on MI350) - distributes work across chiplets
2. L2 Cache Grouping (GROUP_M parameter)
3. Use matrix_instr_nonkdim=16 for 16x16 MFMA instructions
4. Precompute grid/strides in __init__, not forward()
5. Enable TRITON_HIP_USE_BLOCK_PINGPONG and TRITON_HIP_USE_ASYNC_COPY

Output ONLY the optimized Python code in ```python ... ``` block.
'''


def build_reflection_prompt(original_prompt: str, generated_code: str, error_message: str, retry_count: int) -> str:
    """Construct a reflection prompt with error feedback for LLM to retry generation.
    
    Args:
        original_prompt: The original prompt used for generation
        generated_code: The code that was generated but failed
        error_message: The error message from compilation or correctness check
        retry_count: Current retry attempt number
    
    Returns:
        A new prompt that includes the original request, previous attempt, and error feedback
    """
    reflection_prompt = f"""The previous attempt to generate the kernel failed. Here is the feedback:

ORIGINAL REQUEST:
{original_prompt}

PREVIOUS GENERATED CODE (Attempt #{retry_count}):
```python
{generated_code}
```

ERROR MESSAGE:
{error_message}

Please analyze the error and generate a corrected version of the kernel that fixes the issues mentioned above. Make sure to:
1. Carefully read the error message and understand what went wrong
2. Fix the compilation errors or correctness issues
3. Generate complete, working code that passes all tests
4. Follow the same requirements from the original request

Generate the corrected kernel code:"""
    
    return reflection_prompt


def build_optimization_prompt(original_prompt: str, ref_code: str, custom_kernel: str, speedup: float, retry_count: int) -> str:
    """Construct an optimization prompt when the generated kernel is slower than reference.
    
    Args:
        original_prompt: The original prompt used for generation
        ref_code: The reference implementation code
        custom_kernel: The generated kernel code that is correct but slow
        speedup: The speedup ratio (< 1.0 means slower than reference)
        retry_count: Current retry attempt number
    
    Returns:
        A new prompt that asks LLM to optimize the kernel for better performance
    """
    # Detect if this is an element-wise operation
    is_elementwise = any(op in ref_code.lower() for op in [
        'sigmoid', 'relu', 'gelu', 'tanh', 'swish', 'silu', 'softplus',
        'leaky_relu', 'elu', 'selu', 'mish', 'hardswish', 'hardsigmoid'
    ]) or ('* x' in ref_code and 'mm(' not in ref_code and 'matmul' not in ref_code)
    
    is_gemm = any(op in ref_code.lower() for op in [
        'mm(', 'matmul(', 'bmm(', 'linear(', 'nn.linear'
    ])
    
    if is_elementwise and not is_gemm:
        optimization_hints = """
For element-wise Triton kernels (activation functions, etc.) on AMD MI350:

**CRITICAL: Do NOT use internal loops for element-wise operations!**
This is a common mistake that destroys performance.

**WRONG (EXTREMELY SLOW)**:
```python
for i in range(0, BLOCK_SIZE, 8):  # Don't do this!
    offsets = block_start + i + tl.arange(0, 8)
    x = tl.load(x_ptr + offsets, mask=mask)
```

**CORRECT (FAST)**:
```python
# Process entire block at once - no internal loop!
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
y = operation(x)  # Apply operation
tl.store(output_ptr + offsets, y, mask=mask)
```

**Element-wise optimization strategies:**
1. Use larger BLOCK_SIZE (1024-2048) for better memory throughput
2. Use float32 for computation, convert back for storage
3. Precompute grid size in __init__ to reduce launch overhead
4. Use preallocated output buffer if input size is fixed
5. Do NOT use XCD swizzle - it's only for GEMM operations!
6. Ensure coalesced memory access (sequential threads access sequential memory)

**Enable MI350 optimizations:**
```python
os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'
```"""
    else:
        optimization_hints = """
For Triton kernels on AMD MI350 (gfx950), apply these optimization strategies:

1. **XCD Swizzle (CRITICAL for MI350)**: MI350 has 32 XCDs (chiplets). Use this pattern:
```python
NUM_XCDS = 32
pid = tl.program_id(0)
num_pids = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
pids_per_xcd = (num_pids + NUM_XCDS - 1) // NUM_XCDS
xcd_id = pid % NUM_XCDS
local_pid = pid // NUM_XCDS
new_pid = xcd_id * pids_per_xcd + local_pid
```

2. **L2 Cache Grouping**: Use GROUP_M parameter for better L2 cache reuse:
```python
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

3. **MFMA Instructions**: Use matrix_instr_nonkdim=16 for 16x16 MFMA on MI350:
```python
kernel[grid](... , num_warps=8, matrix_instr_nonkdim=16)
```

4. **Launch Overhead Elimination**: Precompute in __init__, not forward():
```python
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Precompute constants
        self.M, self.N, self.K = 4096, 4096, 4096
        # Precompute grid
        self._grid = lambda meta: (triton.cdiv(self.M, meta['BLOCK_M']) * triton.cdiv(self.N, meta['BLOCK_N']),)
```

5. **Environment Variables**: Set these for optimal MI350 performance:
```python
os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'
```

6. **@triton.autotune** for block sizes: Use configs optimized for MI350:
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
```

7. **Vectorized Memory Access**: Use appropriate vector widths for bf16/fp16."""

    optimization_prompt = f"""The previous kernel generation was CORRECT but too SLOW (speedup: {speedup:.3f}x, meaning it's {1/speedup:.2f}x slower than reference). 
We need to optimize it for better performance.

ORIGINAL REQUEST:
{original_prompt}

REFERENCE IMPLEMENTATION:
```python
{ref_code}
```

CURRENT GENERATED KERNEL (Attempt #{retry_count}, Correct but Slow):
```python
{custom_kernel}
```

PERFORMANCE ISSUE:
The kernel is functionally correct but achieves only {speedup:.3f}x speedup (current runtime is {1/speedup:.2f}x SLOWER than reference).
We need speedup > 1.0x to be faster than the reference implementation.

{optimization_hints}

Please analyze the performance bottleneck and generate an OPTIMIZED version that:
1. Maintains correctness (passes all tests)
2. Achieves speedup > 1.0x (faster than reference)
3. Implements the optimization strategies mentioned above
4. Uses hardware-specific optimizations for AMD MI350

You MUST guarantee the correctness of ModelNew. Do NOT cheat by simplifying logic or skipping computations. Do NOT directly and totally use PyTorch native operators to implement the forward function.

Generate the optimized kernel code:"""
    
    return optimization_prompt


def build_optimize_prompt(code: str, feedback: str = None, new_class_name: str = None) -> str:
    """Build user prompt for optimization with optional feedback."""
    prompt = f'''**Triton Kernel to Optimize:**
```python
{code}
```

Please optimize this Triton kernel for AMD MI350 (gfx950) GPU.
Apply all missing MI350 optimizations and return the complete optimized code.
'''
    
    # Specify new class name if provided
    if new_class_name:
        prompt += f'''
**IMPORTANT: Name the optimized model class `{new_class_name}`** (not ModelNew).
'''
    
    if feedback:
        prompt += f'''

**IMPORTANT - Feedback from Profiler/Evaluation:**
{feedback}

Please address the issues mentioned above and generate improved code.
'''
    
    return prompt


def rename_class_in_code(code: str, old_name: str, new_name: str) -> str:
    """Rename a class in the generated code."""
    import re
    # Replace class definition
    code = re.sub(rf'\bclass\s+{old_name}\s*\(', f'class {new_name}(', code)
    # Replace any references to the class
    code = re.sub(rf'\b{old_name}\s*\(', f'{new_name}(', code)
    return code


def main():
    parser = argparse.ArgumentParser(description="Generate HipKittens/Triton kernel from KernelBench problem")
    parser.add_argument("--problem", default=None, help="Path to KernelBench problem file")
    parser.add_argument("--optimize", default=None, help="Path to Triton code file to optimize (alternative mode)")
    parser.add_argument("--output", required=True, help="Output file path for generated code")
    parser.add_argument("--prompt", default=None, help="Path to system prompt file (optional)")
    parser.add_argument("--response-file", default=None, help="Save full LLM response to this file")
    parser.add_argument("--use-template", action="store_true", help="Use pre-verified template for GEMM")
    parser.add_argument("--num-samples", type=int, default=3, 
                        help="Number of code samples to generate in parallel (default: 3)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Base temperature for LLM sampling (default: 0.1)")
    parser.add_argument("--backend", choices=BACKENDS, default="hip",
                        help="Backend type: 'hip' for HipKittens, 'triton' for Triton (default: hip)")
    parser.add_argument("--feedback", default=None, help="Feedback from previous optimization attempt")
    parser.add_argument("--new-class-name", default=None, help="Name for the new generated model class")
    # Retry mode arguments (for reflection/optimization loop)
    parser.add_argument("--retry-mode", choices=["reflection", "optimization"], default=None,
                        help="Retry mode: 'reflection' for error feedback, 'optimization' for perf improvement")
    parser.add_argument("--prev-code", default=None, help="Path to previous generated code (for retry)")
    parser.add_argument("--ref-code", default=None, help="Path to reference code (for optimization)")
    parser.add_argument("--error-msg", default=None, help="Error message from previous attempt (for reflection)")
    parser.add_argument("--speedup", type=float, default=None, help="Previous speedup value (for optimization)")
    parser.add_argument("--attempt", type=int, default=1, help="Current attempt number")
    args = parser.parse_args()
    
    # Validate arguments
    if not args.problem and not args.optimize and not args.retry_mode:
        print("Error: Either --problem, --optimize, or --retry-mode must be specified")
        sys.exit(1)
    
    # Handle retry mode (reflection or optimization)
    if args.retry_mode:
        print(f"Retry mode: {args.retry_mode} (attempt #{args.attempt})")
        
        # Check LLM_GATEWAY_KEY
        if not os.environ.get("LLM_GATEWAY_KEY"):
            print("Error: LLM_GATEWAY_KEY environment variable not set")
            sys.exit(1)
        
        # Load required files
        if not args.prev_code or not args.problem:
            print("Error: --prev-code and --problem are required for retry mode")
            sys.exit(1)
        
        with open(args.prev_code) as f:
            prev_code = f.read()
        
        with open(args.problem) as f:
            problem_code = f.read()
        
        # Build the appropriate prompt based on retry mode
        if args.retry_mode == "reflection":
            # Reflection mode: fix errors
            error_msg = args.error_msg or "Unknown error"
            print(f"  Error to fix: {error_msg[:100]}...")
            
            # Load base system prompt
            problem_name = os.path.basename(args.problem)
            system_prompt = load_prompt(args.prompt, problem_name, backend=args.backend)
            
            # Build reflection prompt
            original_prompt = f"Generate a Triton kernel for the following PyTorch code:\n```python\n{problem_code}\n```"
            user_prompt = build_reflection_prompt(original_prompt, prev_code, error_msg, args.attempt)
            
        else:  # optimization mode
            # Optimization mode: improve performance
            speedup = args.speedup or 0.5
            print(f"  Current speedup: {speedup:.3f}x (target: >= 1.0x)")
            
            # Load ref code if provided, otherwise use problem code
            if args.ref_code:
                with open(args.ref_code) as f:
                    ref_code = f.read()
            else:
                ref_code = problem_code
            
            # Load optimization system prompt
            system_prompt = load_optimize_prompt()
            
            # Build optimization prompt
            original_prompt = f"Generate a Triton kernel for the following PyTorch code:\n```python\n{problem_code}\n```"
            user_prompt = build_optimization_prompt(original_prompt, ref_code, prev_code, speedup, args.attempt)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Calling LLM for {args.retry_mode}...")
        response = call_llm(system_prompt, user_prompt, temperature=args.temperature)
        
        # Save response if requested
        if args.response_file:
            with open(args.response_file, 'w') as f:
                f.write(response)
            print(f"Full response saved to: {args.response_file}")
        
        # Extract and save code
        generated_code = extract_code(response)
        
        # Ensure the class name is correct
        if args.new_class_name:
            generated_code = rename_class_in_code(generated_code, 'ModelNew', args.new_class_name)
        
        with open(args.output, 'w') as f:
            f.write(generated_code)
        
        print(f"Generated code saved to: {args.output}")
        print(f"Code length: {len(generated_code)} characters")
        return
    
    # Handle optimization mode
    if args.optimize:
        print(f"Optimization mode: {args.optimize}")
        if args.feedback:
            print(f"With feedback: {args.feedback[:100]}...")
        if args.new_class_name:
            print(f"New class name: {args.new_class_name}")
        
        # Check LLM_GATEWAY_KEY
        if not os.environ.get("LLM_GATEWAY_KEY"):
            print("Error: LLM_GATEWAY_KEY environment variable not set")
            sys.exit(1)
        
        # Load code to optimize
        with open(args.optimize) as f:
            code_to_optimize = f.read()
        
        # Load optimization prompt
        system_prompt = load_optimize_prompt()
        user_prompt = build_optimize_prompt(
            code_to_optimize, 
            feedback=args.feedback,
            new_class_name=args.new_class_name
        )
        
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        
        print("Calling LLM for optimization...")
        response = call_llm(system_prompt, user_prompt, temperature=args.temperature)
        
        # Save response if requested
        if args.response_file:
            with open(args.response_file, 'w') as f:
                f.write(response)
            print(f"Full response saved to: {args.response_file}")
        
        # Extract and save optimized code
        optimized_code = extract_code(response)
        
        # Ensure the class name is correct (LLM might still use ModelNew)
        if args.new_class_name:
            # Replace ModelNew with the requested class name
            optimized_code = rename_class_in_code(optimized_code, 'ModelNew', args.new_class_name)
        
        with open(args.output, 'w') as f:
            f.write(optimized_code)
        
        print(f"Optimized code saved to: {args.output}")
        print(f"Code length: {len(optimized_code)} characters")
        return
    
    # Load problem
    print(f"Loading problem: {args.problem}")
    problem = load_problem(args.problem)
    
    # Classify problem type for prompt selection
    problem_type = classify_problem(problem["name"], problem["code"])
    print(f"Problem type: {problem_type}")
    print(f"Backend: {args.backend}")
    
    # Check LLM_GATEWAY_KEY for non-template cases
    if not os.environ.get("LLM_GATEWAY_KEY"):
        print("Error: LLM_GATEWAY_KEY environment variable not set")
        sys.exit(1)
    
    # Load prompt (use problem name to select appropriate prompt from config)
    system_prompt = load_prompt(args.prompt, problem["name"], backend=args.backend)
    user_prompt = build_user_prompt(problem, backend=args.backend)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples
    if args.num_samples > 1:
        print(f"Generating {args.num_samples} samples in parallel (temp={args.temperature}, backend={args.backend})...")
        samples = generate_samples(system_prompt, user_prompt, 
                                   num_samples=args.num_samples, 
                                   temperature=args.temperature,
                                   backend=args.backend)
        
        # Save all samples
        output_base = args.output.rsplit('.', 1)[0]  # Remove .py extension
        output_ext = args.output.rsplit('.', 1)[1] if '.' in args.output else 'py'
        
        sample_paths = []
        for sample in samples:
            if sample["code"]:
                sample_path = f"{output_base}_s{sample['id']}.{output_ext}"
                with open(sample_path, 'w') as f:
                    f.write(sample["code"])
                sample_paths.append(sample_path)
                print(f"  Sample {sample['id']}: {len(sample['code'])} chars -> {sample_path}")
                
                # Save response if requested
                if args.response_file:
                    resp_path = f"{args.response_file.rsplit('.', 1)[0]}_s{sample['id']}.txt"
                    with open(resp_path, 'w') as f:
                        f.write(sample["response"] or "")
            else:
                print(f"  Sample {sample['id']}: FAILED - {sample['error']}")
        
        # Also save the first successful sample as the main output
        for sample in samples:
            if sample["code"]:
                with open(args.output, 'w') as f:
                    f.write(sample["code"])
                print(f"\nMain output saved to: {args.output}")
                break
        
        # Output sample paths for downstream processing
        print(f"\nGenerated {len(sample_paths)} samples:")
        for path in sample_paths:
            print(f"  - {path}")
            
    else:
        # Single sample mode (original behavior)
        print("Calling LLM...")
        response = call_llm(system_prompt, user_prompt, temperature=args.temperature)
        
        # Save response if requested
        if args.response_file:
            with open(args.response_file, 'w') as f:
                f.write(response)
            print(f"Full response saved to: {args.response_file}")
        
        # Extract and save code
        code = extract_code(response)
        
        with open(args.output, 'w') as f:
            f.write(code)
        
        print(f"Generated code saved to: {args.output}")
        print(f"Code length: {len(code)} characters")


if __name__ == "__main__":
    main()

