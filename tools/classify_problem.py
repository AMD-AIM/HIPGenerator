#!/usr/bin/env python3
"""
LLM-based problem type classification tool.
Uses LLM to analyze PyTorch code and determine the problem type for optimal prompt selection.
"""
import os
import sys
import json
import re
from pathlib import Path
from typing import Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from anthropic import Anthropic
except ImportError:
    print("Please install anthropic: pip install anthropic")
    sys.exit(1)


# Valid problem types and their descriptions
PROBLEM_TYPES = {
    "gemm": "Matrix multiplication (GEMM, matmul, linear layers, batch matmul)",
    "elementwise": "Element-wise operations (activation functions like ReLU, GELU, Sigmoid, Tanh, Swish, etc.)",
    "softmax": "Softmax and log-softmax operations",
    "norm": "Normalization layers (LayerNorm, BatchNorm, RMSNorm, GroupNorm, InstanceNorm)",
    "reduction": "Reduction operations (sum, mean, max, min over dimensions)",
    "pooling": "Pooling operations (MaxPool, AvgPool, AdaptivePool)",
    "attention": "Attention mechanisms (self-attention, cross-attention, multi-head attention)",
    "conv": "Convolution operations (Conv1d, Conv2d, Conv3d)",
    "embedding": "Embedding operations (lookup, positional encoding)",
    "loss": "Loss functions (CrossEntropy, MSE, etc.)",
    "unknown": "Unknown or complex composite operations"
}


def get_llm_client():
    """Get Anthropic client configured for AMD LLM Gateway."""
    api_key = os.environ.get('LLM_GATEWAY_KEY')
    if not api_key:
        raise ValueError("LLM_GATEWAY_KEY environment variable not set")
    
    return Anthropic(
        api_key=api_key,
        base_url="https://llm-api.amd.com/Anthropic",
        default_headers={
            "Ocp-Apim-Subscription-Key": api_key
        }
    )


def classify_with_llm(problem_code: str, problem_name: str) -> Tuple[str, str]:
    """
    Use LLM to classify the problem type.
    
    Returns:
        Tuple of (problem_type, explanation)
    """
    client = get_llm_client()
    
    # Build classification prompt
    types_list = "\n".join([f"- {k}: {v}" for k, v in PROBLEM_TYPES.items()])
    
    prompt = f"""Analyze the following PyTorch code and classify it into ONE of these problem types:

{types_list}

Code to analyze:
```python
{problem_code}
```

Problem name: {problem_name}

Respond with ONLY a JSON object in this exact format:
{{"type": "<one of the types above>", "reason": "<brief 1-sentence explanation>"}}

Focus on the MAIN computational operation in the forward() method. If there are multiple operations, classify based on the dominant/most expensive one."""

    response = client.messages.create(
        model="claude-opus-4",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse response
    response_text = response.content[0].text.strip()
    
    # Try to extract JSON
    try:
        # Handle markdown code blocks
        if "```" in response_text:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
        
        result = json.loads(response_text)
        problem_type = result.get("type", "unknown").lower()
        reason = result.get("reason", "")
        
        # Validate type
        if problem_type not in PROBLEM_TYPES:
            # Try to map to closest type
            type_lower = problem_type.lower()
            for valid_type in PROBLEM_TYPES:
                if valid_type in type_lower or type_lower in valid_type:
                    problem_type = valid_type
                    break
            else:
                problem_type = "unknown"
        
        return problem_type, reason
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to parse LLM response: {e}")
        print(f"Response was: {response_text}")
        return fallback_classify(problem_code, problem_name), "Fallback classification"


def fallback_classify(problem_code: str, problem_name: str) -> str:
    """
    Fallback classification using pattern matching when LLM fails.
    """
    code_lower = problem_code.lower()
    name_lower = problem_name.lower()
    
    # Check patterns
    patterns = [
        (["relu", "gelu", "swish", "silu", "sigmoid", "tanh", "leaky", "elu", "selu", "mish"], "elementwise"),
        (["softmax", "logsoftmax"], "softmax"),
        (["layernorm", "batchnorm", "rmsnorm", "groupnorm", "instancenorm", "_norm"], "norm"),
        (["pool", "maxpool", "avgpool", "adaptivepool"], "pooling"),
        (["sum_reduction", "mean_reduction", "reduce"], "reduction"),
        (["attention", "multihead", "self_attention"], "attention"),
        (["conv1d", "conv2d", "conv3d", "convolution"], "conv"),
        (["embedding", "positional"], "embedding"),
        (["loss", "crossentropy", "mse", "bce"], "loss"),
        (["matmul", "mm", "linear", "gemm", "matrix_mult"], "gemm"),
    ]
    
    for keywords, ptype in patterns:
        for kw in keywords:
            if kw in name_lower or kw in code_lower:
                return ptype
    
    # Check for common operations in code
    if "torch.matmul" in code_lower or "torch.mm(" in code_lower or "@" in problem_code:
        return "gemm"
    if "torch.softmax" in code_lower or "F.softmax" in code_lower:
        return "softmax"
    
    return "unknown"


def classify_problem(problem_path: str, use_llm: bool = True) -> dict:
    """
    Classify a problem file.
    
    Args:
        problem_path: Path to the problem Python file
        use_llm: Whether to use LLM for classification (default True)
        
    Returns:
        Dict with type, reason, and prompt recommendation
    """
    with open(problem_path) as f:
        problem_code = f.read()
    
    problem_name = Path(problem_path).stem
    
    if use_llm:
        try:
            problem_type, reason = classify_with_llm(problem_code, problem_name)
        except Exception as e:
            print(f"LLM classification failed: {e}, using fallback")
            problem_type = fallback_classify(problem_code, problem_name)
            reason = "Fallback pattern matching"
    else:
        problem_type = fallback_classify(problem_code, problem_name)
        reason = "Pattern matching (LLM disabled)"
    
    # Recommend prompt based on type
    prompt_map = {
        "gemm": "triton_gemm_mi350_golden.txt",
        "elementwise": "triton_elementwise_optimized.txt",
        "softmax": "triton_softmax.txt",
        "norm": "triton_norm.txt",
        "reduction": "triton_reduction.txt",
        "pooling": "triton_pooling.txt",
        "attention": "triton_high_performance.txt",
        "conv": "triton_high_performance.txt",
        "unknown": "triton_high_correctness.txt",
    }
    
    recommended_prompt = prompt_map.get(problem_type, "triton_high_correctness.txt")
    
    return {
        "problem_name": problem_name,
        "problem_type": problem_type,
        "type_description": PROBLEM_TYPES.get(problem_type, "Unknown"),
        "reason": reason,
        "recommended_prompt": recommended_prompt
    }


def main():
    """CLI interface for problem classification."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify problem type using LLM")
    parser.add_argument("problem", help="Path to problem file")
    parser.add_argument("--no-llm", action="store_true", help="Use pattern matching only")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if not Path(args.problem).exists():
        print(f"Error: File not found: {args.problem}")
        sys.exit(1)
    
    result = classify_problem(args.problem, use_llm=not args.no_llm)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Problem:     {result['problem_name']}")
        print(f"Type:        {result['problem_type']}")
        print(f"Description: {result['type_description']}")
        print(f"Reason:      {result['reason']}")
        print(f"Prompt:      {result['recommended_prompt']}")


if __name__ == "__main__":
    main()

