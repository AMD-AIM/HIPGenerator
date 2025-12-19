#!/bin/bash
# Batch test HipKittens kernel generation on KernelBench level1 problems
# Usage: ./run_batch.sh [start_id] [end_id]

set -e

# Disable core dumps to avoid large files from GPU crashes
ulimit -c 0

# Set LLM API key if not already set
export LLM_GATEWAY_KEY="${LLM_GATEWAY_KEY:-}"
if [[ -z "${LLM_GATEWAY_KEY}" ]]; then
    echo "Error: LLM_GATEWAY_KEY is not set. Export it before running." >&2
    echo "Example: export LLM_GATEWAY_KEY='...'" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNELBENCH_DIR="${KERNELBENCH_DIR:-/root/agent/kernel-agent/datasets/KernelBench/level1}"
RESULTS_DIR="${SCRIPT_DIR}/results"
MAX_RETRIES=3
NUM_SAMPLES=${NUM_SAMPLES:-3}  # Number of parallel samples per attempt

# Parse arguments
START_ID=${1:-1}
END_ID=${2:-40}

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Initialize summary file
SUMMARY_FILE="${RESULTS_DIR}/summary.json"
echo '{"tasks": []}' > "${SUMMARY_FILE}"

# Log file
LOG_FILE="${RESULTS_DIR}/batch_run.log"
echo "Starting batch run at $(date)" | tee "${LOG_FILE}"
echo "Testing problems ${START_ID} to ${END_ID}" | tee -a "${LOG_FILE}"

PROMPTS_DIR="${SCRIPT_DIR}/prompts"
PROMPT_CONFIG="${PROMPTS_DIR}/config.json"

# Function to get prompt file based on problem name
get_prompt_for_problem() {
    local problem_name="$1"
    local problem_lower=$(echo "$problem_name" | tr '[:upper:]' '[:lower:]')
    
    # Read patterns from config and find matching prompt
    if [[ -f "${PROMPT_CONFIG}" ]]; then
        export PROBLEM_NAME_LOWER="${problem_lower}"
        export CONFIG_PATH="${PROMPT_CONFIG}"
        local prompt_file=$(python3 << 'PYEOF'
import json
import re
import sys
import os

config = json.load(open(os.environ["CONFIG_PATH"]))
problem = os.environ["PROBLEM_NAME_LOWER"]

for pattern, prompt in config.get("patterns", {}).items():
    if re.search(pattern, problem, re.IGNORECASE):
        print(prompt)
        sys.exit(0)

print(config.get("default_prompt", "elementwise_bf16.txt"))
PYEOF
)
        echo "${PROMPTS_DIR}/${prompt_file}"
    else
        echo "${PROMPTS_DIR}/elementwise_bf16.txt"
    fi
}

# Function to get reflection hints from config
get_reflection_hints() {
    local error_type="$1"
    
    if [[ -f "${PROMPT_CONFIG}" ]]; then
        python3 << 'PYEOF'
import json
import sys
import os

config_path = os.environ.get("PROMPT_CONFIG_PATH", "")
error_type = os.environ.get("ERROR_TYPE", "")

if config_path and os.path.exists(config_path):
    config = json.load(open(config_path))
    hints = config.get("reflection_hints", {})
    
    for key, values in hints.items():
        if key in error_type:
            for hint in values[:3]:
                print(f"- {hint}")
            break
PYEOF
    fi
}
export PROMPT_CONFIG_PATH="${PROMPT_CONFIG}"

# Function to update prompt based on error - implements reflection
create_improved_prompt() {
    local error_msg="$1"
    local attempt="$2"
    local prompt_file="$3"
    local prev_code_file="$4"
    local problem_name="$5"
    
    if [[ "${attempt}" == "1" ]]; then
        # First attempt: use prompt from config based on problem type
        local base_prompt=$(get_prompt_for_problem "${problem_name}")
        if [[ -f "${base_prompt}" ]]; then
            cp "${base_prompt}" "${prompt_file}"
        else
            # Fallback to default
            cp "${PROMPTS_DIR}/elementwise_bf16.txt" "${prompt_file}"
        fi
    else
        # Subsequent attempts: include previous error and code for reflection
        local base_prompt=$(get_prompt_for_problem "${problem_name}")
        
        # Extract the most relevant part of error (compiler messages are at the end)
        local error_len=${#error_msg}
        local error_excerpt
        if [[ $error_len -gt 4000 ]]; then
            # Keep last 4000 chars - contains actual compiler errors
            error_excerpt="${error_msg: -4000}"
        else
            error_excerpt="${error_msg}"
        fi
        
        cat > "${prompt_file}" << PROMPT_END
You are an expert C++/HIP programmer for AMD GPUs.

**PREVIOUS ATTEMPT FAILED with this COMPILER ERROR:**
\`\`\`
${error_excerpt}
\`\`\`

**ANALYZE THE ERROR ABOVE CAREFULLY:**
- Look for specific line numbers and error messages
- Common issues: wrong HipKittens API usage, missing includes, type mismatches

**PREVIOUS CODE (fix the issues):**
\`\`\`python
$(cat "${prev_code_file}" 2>/dev/null | head -200)
\`\`\`

**FIX THE SPECIFIC ERRORS SHOWN ABOVE! Key fixes:**
PROMPT_END

        # Add reflection hints from config
        export ERROR_TYPE="${error_msg:0:200}"
        get_reflection_hints >> "${prompt_file}"
        
        # Add specific fixes based on error patterns
        if [[ "$error_msg" == *"PERFORMANCE TOO SLOW"* ]]; then
            cat >> "${prompt_file}" << 'PERF_HINT'

**PERFORMANCE OPTIMIZATION REQUIRED:**
- Use float4 for 128-bit vectorized loads (8 bf16 per load)
- NEVER use .to() for type conversion - it's extremely slow!
- Process bf16 directly using hip_bfloat16
- Add #pragma unroll for inner loops
PERF_HINT
        fi
        
        # Add rocprof metrics if available
        if [[ -f "${task_dir}/result_$((attempt-1)).json" ]]; then
            local rocprof_hints=$(python3 -c "
import json, sys
try:
    r = json.load(open('${task_dir}/result_$((attempt-1)).json'))
    m = r.get('rocprof_metrics', {})
    hints = m.get('optimization_hints', [])
    raw = m.get('raw_metrics', {})
    lds_kb = raw.get('lds_kb', 0)
    speedup = r.get('speedup', 0)
    
    print(f'Previous attempt speedup: {speedup:.2f}x')
    
    if lds_kb > 100:
        print(f'- CRITICAL: LDS usage {lds_kb:.1f}KB severely limits occupancy (1 wave/CU)')
        print(f'  To reduce LDS, consider:')
        print(f'  * Reduce BLOCK_SIZE from 256 to 128')
        print(f'  * Use smaller shared tiles (st_bf<64,64> instead of st_bf<128,64>)')
        print(f'  * Target LDS < 48KB for 4+ concurrent waves')
    elif lds_kb > 48:
        print(f'- HIGH LDS USAGE: {lds_kb:.1f}KB limits occupancy. Target < 48KB.')
    
    for h in hints:
        if 'LDS' not in h:  # Avoid duplicate LDS hints
            print(f'- {h}')
except Exception as e:
    pass
" 2>/dev/null || true)
            if [[ -n "$rocprof_hints" ]]; then
                echo "" >> "${prompt_file}"
                echo "**ROCPROF ANALYSIS (from previous attempt):**" >> "${prompt_file}"
                echo "$rocprof_hints" >> "${prompt_file}"
            fi
        fi
        
        if [[ "$error_msg" == *"hip_sources"* ]]; then
            echo "- ERROR: Used hip_sources. FIX: Use cuda_sources instead!" >> "${prompt_file}"
        fi
        
        if [[ "$error_msg" == *"No custom kernels"* ]]; then
            echo "- ERROR: No custom kernel detected! You MUST implement a HIP kernel." >> "${prompt_file}"
        fi
        
        # Append the base template
        echo "" >> "${prompt_file}"
        echo "**FOLLOW THIS TEMPLATE:**" >> "${prompt_file}"
        if [[ -f "${base_prompt}" ]]; then
            cat "${base_prompt}" >> "${prompt_file}"
        fi
        
        if [[ "$error_msg" == *"PYBIND11"* ]] || [[ "$error_msg" == *"redefinition"* ]] || [[ "$error_msg" == *"multiple definition"* ]]; then
            echo "- ERROR: PYBIND11_MODULE in cpp_src. FIX: Remove it! load_inline adds it automatically!" >> "${prompt_file}"
        fi
        
        if [[ "$error_msg" == *"undeclared identifier"* ]]; then
            echo "- ERROR: Undeclared identifier. FIX: Check includes and variable names!" >> "${prompt_file}"
        fi
        
        if [[ "$error_msg" == *"mma_ABt"* ]] || [[ "$error_msg" == *"constraints not satisfied"* ]]; then
            echo "- ERROR: HipKittens mma_ABt is complex. FIX: Use torch.mm() for GEMM instead!" >> "${prompt_file}"
        fi
        
        if [[ "$error_msg" == *"no member"* ]] || [[ "$error_msg" == *"no type"* ]]; then
            echo "- ERROR: Wrong API usage. FIX: Use simple HIP kernels with float type!" >> "${prompt_file}"
        fi
        
        if [[ "$error_msg" == *"nan"* ]] || [[ "$error_msg" == *"inf"* ]] || [[ "$error_msg" == *"Inf"* ]]; then
            echo "- ERROR: NaN/Inf values. FIX: Check array bounds and initialization!" >> "${prompt_file}"
        fi

        cat >> "${prompt_file}" << 'PROMPT_END'

**WORKING TEMPLATE (use this!):**
```python
from torch.utils.cpp_extension import load_inline

cpp_src = '''
#include <torch/extension.h>
torch::Tensor forward_func(torch::Tensor x);
'''

hip_src = '''
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>

__global__ void my_kernel(const float* in, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = /* your computation */;
    }
}

torch::Tensor forward_func(torch::Tensor x) {
    auto x_f32 = x.to(torch::kFloat32).contiguous();
    auto output = torch::empty_like(x_f32);
    int64_t size = x_f32.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    my_kernel<<<blocks, threads>>>(x_f32.data_ptr<float>(), output.data_ptr<float>(), size);
    return output.to(x.dtype());
}
'''

module = load_inline(
    name="my_module",
    cpp_sources=cpp_src,
    cuda_sources=hip_src,
    functions=["forward_func"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "-std=c++20", "--offload-arch=gfx950"],
    verbose=False
)
```

Generate FIXED Python code. Output ONLY code, no explanations.
PROMPT_END
    fi
}

# Function to test a single problem
test_problem() {
    local problem_id="$1"
    local problem_file="${KERNELBENCH_DIR}/${problem_id}_*.py"
    
    # Find the actual file
    local actual_file=$(ls ${problem_file} 2>/dev/null | head -1)
    if [[ ! -f "${actual_file}" ]]; then
        echo "Problem file not found: ${problem_file}" | tee -a "${LOG_FILE}"
        return 1
    fi
    
    local problem_name=$(basename "${actual_file}" .py)
    local task_dir="${RESULTS_DIR}/${problem_name}"
    mkdir -p "${task_dir}"
    
    echo "" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "Testing: ${problem_name}" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    
    local best_speedup=0
    local best_attempt=0
    local success=false
    local last_error=""
    
    for attempt in $(seq 1 ${MAX_RETRIES}); do
        echo "Attempt ${attempt}/${MAX_RETRIES}..." | tee -a "${LOG_FILE}"
        
        local prompt_file="${task_dir}/prompt_${attempt}.txt"
        local code_file="${task_dir}/code_${attempt}.py"
        local result_file="${task_dir}/result_${attempt}.json"
        local response_file="${task_dir}/response_${attempt}.txt"
        
        # Get previous code file for reflection
        local prev_code_file=""
        if [[ ${attempt} -gt 1 ]]; then
            prev_code_file="${task_dir}/code_$((attempt-1)).py"
        fi
        
        # Create/update prompt with reflection
        create_improved_prompt "${last_error}" "${attempt}" "${prompt_file}" "${prev_code_file}" "${problem_name}"
        
        # Generate code (multiple samples in parallel)
        local num_samples=${NUM_SAMPLES:-3}
        echo "  Generating ${num_samples} code samples..." | tee -a "${LOG_FILE}"
        python3 "${SCRIPT_DIR}/generate.py" \
            --problem "${actual_file}" \
            --output "${code_file}" \
            --prompt "${prompt_file}" \
            --response-file "${response_file}" \
            --num-samples "${num_samples}" \
            2>&1 | tee -a "${LOG_FILE}" || {
                last_error="Generation failed"
                continue
            }
        
        # Clear torch cache (important for fresh compilation)
        rm -rf /root/.cache/torch_extensions/py310_cpu/* 2>/dev/null || true
        sync
        sleep 1
        
        # Evaluate all samples and pick the best one
        local best_sample_speedup=0
        local best_sample_file=""
        local best_sample_result=""
        local sample_accuracy_pass=false
        
        # Always run profiler to collect metrics for optimization analysis
        local profile_flag="--profile"
        
        # Find all generated samples
        local code_base="${code_file%.py}"
        local sample_files=$(ls ${code_base}_s*.py 2>/dev/null || echo "${code_file}")
        
        if [[ -z "${sample_files}" ]] || [[ "${sample_files}" == "${code_file}" ]]; then
            # Single sample mode fallback
            sample_files="${code_file}"
        fi
        
        echo "  Evaluating samples..." | tee -a "${LOG_FILE}"
        for sample_file in ${sample_files}; do
            if [[ ! -f "${sample_file}" ]]; then
                continue
            fi
            
            local sample_id=$(echo "${sample_file}" | grep -oP '_s\K[0-9]+' || echo "1")
            local sample_result="${task_dir}/result_${attempt}_s${sample_id}.json"
            
            # Clear torch cache before each sample
            rm -rf /root/.cache/torch_extensions/py310_cpu/* 2>/dev/null || true
            
            local sample_output
            sample_output=$(python3 "${SCRIPT_DIR}/eval.py" \
                --code "${sample_file}" \
                --problem "${actual_file}" \
                --output "${sample_result}" \
                ${profile_flag} 2>&1) || true
            
            # Parse sample result
            if [[ -f "${sample_result}" ]]; then
                local s_speedup=$(python3 -c "import json; print(json.load(open('${sample_result}'))['speedup'])" 2>/dev/null || echo "0")
                local s_accuracy=$(python3 -c "import json; print(json.load(open('${sample_result}'))['accuracy_pass'])" 2>/dev/null || echo "False")
                
                echo "    Sample ${sample_id}: accuracy=${s_accuracy}, speedup=${s_speedup}x" | tee -a "${LOG_FILE}"
                
                if [[ "${s_accuracy}" == "True" ]]; then
                    sample_accuracy_pass=true
                    # Check if this sample is better
                    is_sample_better=$(python3 -c "print('1' if float('${s_speedup:-0}') > float('${best_sample_speedup:-0}') else '0')" 2>/dev/null || echo "0")
                    if [[ "${is_sample_better}" == "1" ]]; then
                        best_sample_speedup="${s_speedup}"
                        best_sample_file="${sample_file}"
                        best_sample_result="${sample_result}"
                    fi
                fi
            fi
        done
        
        # Use the best sample as the main result
        if [[ -n "${best_sample_file}" ]]; then
            cp "${best_sample_file}" "${code_file}"
            cp "${best_sample_result}" "${result_file}"
            echo "  Best sample: $(basename ${best_sample_file}) (speedup=${best_sample_speedup}x)" | tee -a "${LOG_FILE}"
        elif [[ -f "${code_file}" ]]; then
            # No accuracy pass, use first sample for error info
            if [[ -f "${task_dir}/result_${attempt}_s1.json" ]]; then
                cp "${task_dir}/result_${attempt}_s1.json" "${result_file}"
            fi
        fi
        
        local eval_output=""
        local eval_exit_code=0
        
        echo "${eval_output}" | tee -a "${LOG_FILE}"
        
        # Check for runtime crashes in output
        if [[ "${eval_output}" == *"Memory access fault"* ]]; then
            last_error="GPU MEMORY ACCESS FAULT: The kernel crashed due to invalid memory access. This is likely caused by: 1) Unaligned vectorized access (float4/int4 require 16-byte alignment), 2) Out of bounds array access, 3) Incorrect size calculations. Use simpler non-vectorized code or add alignment checks."
            echo "  ✗ GPU crashed" | tee -a "${LOG_FILE}"
            continue
        fi
        
        # Parse result
        if [[ -f "${result_file}" ]]; then
            local speedup=$(python3 -c "import json; print(json.load(open('${result_file}'))['speedup'])" 2>/dev/null || echo "0")
            local accuracy=$(python3 -c "import json; print(json.load(open('${result_file}'))['accuracy_pass'])" 2>/dev/null || echo "False")
            local error=$(python3 -c "import json; e=json.load(open('${result_file}')).get('error',''); print(e[:500] if e else '')" 2>/dev/null || echo "")
            
            if [[ "${accuracy}" == "True" ]]; then
                echo "  ✓ Accuracy passed, speedup: ${speedup}x" | tee -a "${LOG_FILE}"
                
                # Check if this is better (use python for float comparison)
                is_better=$(python3 -c "print('1' if float('${speedup:-0}') > float('${best_speedup:-0}') else '0')" 2>/dev/null || echo "0")
                if [[ "${is_better}" == "1" ]]; then
                    best_speedup="${speedup}"
                    best_attempt="${attempt}"
                    
                    # Copy best files
                    cp "${code_file}" "${task_dir}/best_code.py"
                    cp "${prompt_file}" "${task_dir}/best_prompt.txt"
                    cp "${result_file}" "${task_dir}/best_result.json"
                fi
                
                # If speedup >= 1.0, we're done
                is_done=$(python3 -c "print('1' if float('${speedup:-0}') >= 1.0 else '0')" 2>/dev/null || echo "0")
                if [[ "${is_done}" == "1" ]]; then
                    echo "  ✓✓ Performance target met!" | tee -a "${LOG_FILE}"
                    success=true
                    break
                else
                    # Performance not good enough - provide feedback for next attempt
                    local perf_analysis=""
                    if [[ -f "${result_file}" ]]; then
                        perf_analysis=$(python3 -c "import json; print(json.load(open('${result_file}')).get('perf_analysis',''))" 2>/dev/null || echo "")
                    fi
                    
                    # Check for specific issues
                    local optimization_hint=""
                    if [[ "${perf_analysis}" == *"No custom kernels"* ]] || [[ "${perf_analysis}" == *"PyTorch internals"* ]]; then
                        optimization_hint="CRITICAL: No custom HIP kernels detected! You MUST implement a custom HIP kernel. FORBIDDEN: torch::mm, torch::matmul, torch::mv, torch::bmm. For GEMM use HipKittens MFMA (mma_ABt). For element-wise ops use native HIP kernels."
                    elif [[ "${perf_analysis}" == *"vectorized_elementwise"* ]]; then
                        optimization_hint="CRITICAL: .to() type conversion is the bottleneck! REMOVE ALL .to() calls. Process bf16 directly using hip_bfloat16."
                    else
                        optimization_hint="The code is correct but slow."
                    fi
                    
                    last_error="PERFORMANCE TOO SLOW: speedup=${speedup}x (need >= 1.0x). ${optimization_hint} ${perf_analysis:+Profiler: ${perf_analysis}. }OPTIMIZE: 1) Use HipKittens MFMA for GEMM 2) Use vectorized loads 3) Process bf16 directly 4) Use __restrict__"
                fi
            else
                echo "  ✗ Accuracy failed" | tee -a "${LOG_FILE}"
                last_error="${error:-Accuracy check failed}"
            fi
        else
            last_error="Result file not created"
        fi
    done
    
    # Record final result
    local final_status="failed"
    if [[ "${success}" == "true" ]]; then
        final_status="success"
    else
        local has_result=$(python3 -c "print('1' if float('${best_speedup:-0}') > 0 else '0')" 2>/dev/null || echo "0")
        if [[ "${has_result}" == "1" ]]; then
            final_status="partial"
        fi
    fi
    
    echo "" | tee -a "${LOG_FILE}"
    echo "Final result for ${problem_name}: ${final_status} (best speedup: ${best_speedup}x)" | tee -a "${LOG_FILE}"
    
    # Add to summary
    python3 << EOF
import json
summary = json.load(open('${SUMMARY_FILE}'))
summary['tasks'].append({
    'problem': '${problem_name}',
    'status': '${final_status}',
    'best_speedup': ${best_speedup:-0},
    'best_attempt': ${best_attempt:-0},
    'total_attempts': ${attempt:-0}
})
json.dump(summary, open('${SUMMARY_FILE}', 'w'), indent=2)
EOF
}

# Main loop
for id in $(seq ${START_ID} ${END_ID}); do
    test_problem "${id}" || true
done

# Generate final summary table
echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "FINAL SUMMARY" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

python3 << 'EOF' | tee -a "${LOG_FILE}"
import json
import os

results_dir = os.environ.get('RESULTS_DIR', 'results')
summary_file = os.path.join(results_dir, 'summary.json')

if os.path.exists(summary_file):
    summary = json.load(open(summary_file))
    
    print(f"\n{'Problem':<40} {'Status':<10} {'Speedup':<10} {'Attempts':<10}")
    print("-" * 70)
    
    success_count = 0
    partial_count = 0
    failed_count = 0
    
    for task in summary['tasks']:
        status_icon = {'success': '✓', 'partial': '⚠', 'failed': '✗'}.get(task['status'], '?')
        print(f"{task['problem']:<40} {status_icon} {task['status']:<8} {task['best_speedup']:.2f}x     {task['total_attempts']}")
        
        if task['status'] == 'success':
            success_count += 1
        elif task['status'] == 'partial':
            partial_count += 1
        else:
            failed_count += 1
    
    print("-" * 70)
    print(f"Total: {len(summary['tasks'])} tasks")
    print(f"  ✓ Success (speedup >= 1.0x): {success_count}")
    print(f"  ⚠ Partial (accuracy pass, speedup < 1.0x): {partial_count}")
    print(f"  ✗ Failed: {failed_count}")
EOF

echo "" | tee -a "${LOG_FILE}"
echo "Batch run completed at $(date)" | tee -a "${LOG_FILE}"
echo "Results saved to: ${RESULTS_DIR}" | tee -a "${LOG_FILE}"

