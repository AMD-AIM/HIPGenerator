"""
Evaluator for testing generated Triton kernels.
"""
import os
import sys
import time
import importlib.util
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalResult:
    """Result of kernel evaluation."""
    compile_success: bool = False
    accuracy_pass: bool = False
    speedup: float = 0.0
    ref_time_ms: float = 0.0
    new_time_ms: float = 0.0
    max_diff: float = 0.0
    error: Optional[str] = None
    
    def __str__(self) -> str:
        if not self.compile_success:
            return f"❌ COMPILE FAILED: {self.error}"
        if not self.accuracy_pass:
            return f"❌ ACCURACY FAILED: max_diff={self.max_diff:.6f}"
        return f"✓ PASS: {self.speedup:.2f}x speedup (ref={self.ref_time_ms:.3f}ms, new={self.new_time_ms:.3f}ms)"


class Evaluator:
    """
    Evaluates generated Triton kernels for correctness and performance.
    """
    
    def __init__(
        self,
        warmup_iters: int = 5,
        benchmark_iters: int = 20,
        atol: float = 0.1,
        rtol: float = 0.01,
    ):
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters
        self.atol = atol
        self.rtol = rtol
    
    def evaluate(
        self,
        code_path: str,
        problem_path: str,
    ) -> EvalResult:
        """
        Evaluate a generated kernel against the reference.
        
        Args:
            code_path: Path to generated code
            problem_path: Path to problem definition
            
        Returns:
            EvalResult with evaluation results
        """
        result = EvalResult()
        
        # Load problem module
        try:
            problem_module = self._load_module(problem_path, "problem")
        except Exception as e:
            result.error = f"Failed to load problem: {e}"
            return result
        
        # Load generated code
        try:
            code_module = self._load_module(code_path, "generated")
        except Exception as e:
            result.error = f"Failed to load generated code: {e}"
            return result
        
        # Create models
        try:
            ref_model = problem_module.Model().cuda().eval()
            new_model = code_module.ModelNew().cuda().eval()
            result.compile_success = True
        except Exception as e:
            result.error = f"Failed to create models: {e}"
            return result
        
        # Get inputs
        try:
            inputs = problem_module.get_inputs()
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            # Ensure inputs are on CUDA
            inputs = [x.cuda() if hasattr(x, 'cuda') else x for x in inputs]
        except Exception as e:
            result.error = f"Failed to get inputs: {e}"
            return result
        
        # Test accuracy
        try:
            with torch.no_grad():
                ref_output = ref_model(*inputs)
                new_output = new_model(*inputs)
            
            result.max_diff = self._compute_max_diff(ref_output, new_output)
            result.accuracy_pass = result.max_diff < self.atol
            
            if not result.accuracy_pass:
                result.error = f"Accuracy check failed: max_diff={result.max_diff:.6f}"
                return result
                
        except Exception as e:
            result.error = f"Accuracy test error: {e}"
            result.accuracy_pass = False
            return result
        
        # Benchmark
        try:
            result.ref_time_ms = self._benchmark(ref_model, inputs)
            result.new_time_ms = self._benchmark(new_model, inputs)
            
            if result.ref_time_ms > 0:
                result.speedup = result.ref_time_ms / result.new_time_ms
            
        except Exception as e:
            result.error = f"Benchmark error: {e}"
            # Still return partial results
        
        return result
    
    def _load_module(self, path: str, name: str) -> Any:
        """Load a Python module from path."""
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    
    def _compute_max_diff(self, ref: torch.Tensor, new: torch.Tensor) -> float:
        """Compute maximum difference between tensors."""
        if ref.shape != new.shape:
            return float('inf')
        
        ref_flat = ref.float().flatten()
        new_flat = new.float().flatten()
        
        # Handle NaN/Inf
        if torch.isnan(new_flat).any() or torch.isinf(new_flat).any():
            return float('inf')
        
        diff = torch.abs(ref_flat - new_flat)
        return diff.max().item()
    
    def _benchmark(self, model: torch.nn.Module, inputs: list) -> float:
        """Benchmark a model."""
        # Warmup
        for _ in range(self.warmup_iters):
            with torch.no_grad():
                _ = model(*inputs)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.benchmark_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.benchmark_iters)]
        
        for i in range(self.benchmark_iters):
            start_events[i].record()
            with torch.no_grad():
                _ = model(*inputs)
            end_events[i].record()
        
        torch.cuda.synchronize()
        
        times = [start_events[i].elapsed_time(end_events[i]) for i in range(self.benchmark_iters)]
        return sum(times) / len(times)

