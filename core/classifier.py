"""
Problem type classifier for HIPGenerator.
"""
import re
from pathlib import Path
from typing import Tuple


# Problem type definitions
PROBLEM_TYPES = {
    "gemm": "Matrix multiplication (GEMM, matmul, linear layers)",
    "elementwise": "Element-wise operations (activation functions)",
    "softmax": "Softmax and log-softmax operations",
    "norm": "Normalization layers (LayerNorm, BatchNorm, RMSNorm)",
    "reduction": "Reduction operations (sum, mean, max)",
    "pooling": "Pooling operations (MaxPool, AvgPool)",
    "attention": "Attention mechanisms",
    "conv": "Convolution operations",
    "unknown": "Unknown operation type"
}

# Prompt mapping for each problem type
PROMPT_MAP = {
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


class ProblemClassifier:
    """Classifies problems based on code analysis."""
    
    # Pattern definitions for each type
    PATTERNS = {
        "elementwise": [
            "relu", "gelu", "swish", "silu", "sigmoid", "tanh", 
            "leaky", "elu", "selu", "mish", "hardswish", "hardsigmoid"
        ],
        "softmax": ["softmax", "logsoftmax"],
        "norm": ["layernorm", "batchnorm", "rmsnorm", "groupnorm", "instancenorm"],
        "pooling": ["pool", "maxpool", "avgpool", "adaptivepool"],
        "reduction": ["sum_reduction", "mean_reduction", "max_reduction", "reduce"],
        "attention": ["attention", "multihead", "scaled_dot_product"],
        "conv": ["conv1d", "conv2d", "conv3d"],
        "gemm": ["matmul", "mm(", "bmm(", "linear", "gemm", "matrix_mult"],
    }
    
    def classify(self, problem_path: str) -> Tuple[str, str]:
        """
        Classify a problem based on its code.
        
        Args:
            problem_path: Path to the problem file
            
        Returns:
            Tuple of (problem_type, recommended_prompt)
        """
        with open(problem_path) as f:
            code = f.read()
        
        name = Path(problem_path).stem.lower()
        code_lower = code.lower()
        
        # Check each pattern type
        for ptype, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if pattern in name or pattern in code_lower:
                    return ptype, PROMPT_MAP.get(ptype, PROMPT_MAP["unknown"])
        
        # Fallback checks for code content
        if "torch.matmul" in code_lower or "@" in code:
            return "gemm", PROMPT_MAP["gemm"]
        if "torch.softmax" in code_lower or "F.softmax" in code_lower:
            return "softmax", PROMPT_MAP["softmax"]
        
        return "unknown", PROMPT_MAP["unknown"]
    
    def get_prompt_path(self, problem_type: str, prompts_dir: str = "prompts") -> str:
        """Get the prompt file path for a problem type."""
        prompt_file = PROMPT_MAP.get(problem_type, PROMPT_MAP["unknown"])
        return str(Path(prompts_dir) / prompt_file)
    
    def get_type_description(self, problem_type: str) -> str:
        """Get description for a problem type."""
        return PROBLEM_TYPES.get(problem_type, PROBLEM_TYPES["unknown"])

