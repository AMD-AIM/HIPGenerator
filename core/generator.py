"""
Code generator for Triton kernels using LLM.
"""
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic

from utils.logger import get_logger
from utils.config import get_config
from core.classifier import ProblemClassifier

logger = get_logger(__name__)


class Generator:
    """
    Generates Triton kernel code using LLM.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.classifier = ProblemClassifier()
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize the Anthropic client."""
        api_key = self.config.llm_gateway_key
        if not api_key:
            logger.warning("LLM_GATEWAY_KEY not set")
            return
        
        self.client = Anthropic(
            base_url="https://llm-api.amd.com/Anthropic",
            api_key="dummy",  # Required by client but not used
            default_headers={
                "Ocp-Apim-Subscription-Key": api_key,
            }
        )
    
    def generate(
        self,
        problem_path: str,
        output_path: str,
        num_samples: int = 1,
        temperature: float = 0.1,
    ) -> List[str]:
        """
        Generate Triton kernel code from a PyTorch problem.
        
        Args:
            problem_path: Path to the problem file
            output_path: Path for output code
            num_samples: Number of samples to generate
            temperature: LLM temperature
            
        Returns:
            List of generated code file paths
        """
        if not self.client:
            raise RuntimeError("LLM client not initialized. Set LLM_GATEWAY_KEY.")
        
        # Load problem code
        with open(problem_path) as f:
            problem_code = f.read()
        
        problem_name = Path(problem_path).stem
        
        # Classify problem
        problem_type, prompt_file = self.classifier.classify(problem_path)
        logger.info(f"Problem: {problem_name} | Type: {problem_type}")
        
        # Load system prompt
        system_prompt = self._load_prompt(prompt_file)
        
        # Build user prompt
        user_prompt = self._build_user_prompt(problem_code, problem_type)
        
        # Generate samples
        output_paths = []
        for i in range(num_samples):
            logger.info(f"Generating sample {i+1}/{num_samples}...")
            
            code = self._call_llm(system_prompt, user_prompt, temperature)
            if code:
                # Extract code from response
                code = self._extract_code(code)
                
                # Save code
                if num_samples == 1:
                    out_file = output_path
                else:
                    base = Path(output_path)
                    out_file = str(base.parent / f"{base.stem}_s{i+1}{base.suffix}")
                
                Path(out_file).parent.mkdir(parents=True, exist_ok=True)
                with open(out_file, 'w') as f:
                    f.write(code)
                
                output_paths.append(out_file)
                logger.info(f"Saved: {out_file} ({len(code)} chars)")
        
        return output_paths
    
    def optimize(
        self,
        problem_path: str,
        prev_code_path: str,
        output_path: str,
        speedup: float,
        profiler_feedback: str = "",
    ) -> str:
        """
        Optimize existing Triton kernel code.
        
        Args:
            problem_path: Path to the problem file
            prev_code_path: Path to previous code
            output_path: Path for optimized code
            speedup: Current speedup value
            profiler_feedback: Profiler feedback string
            
        Returns:
            Path to optimized code
        """
        if not self.client:
            raise RuntimeError("LLM client not initialized. Set LLM_GATEWAY_KEY.")
        
        # Load problem and previous code
        with open(problem_path) as f:
            problem_code = f.read()
        with open(prev_code_path) as f:
            prev_code = f.read()
        
        problem_name = Path(problem_path).stem
        problem_type, _ = self.classifier.classify(problem_path)
        
        # Build optimization prompt
        user_prompt = self._build_optimization_prompt(
            problem_code, prev_code, speedup, profiler_feedback, problem_type
        )
        
        # Load system prompt for optimization
        system_prompt = self._load_prompt("triton_high_performance.txt")
        
        logger.info(f"Optimizing {problem_name} (current speedup: {speedup:.2f}x)")
        
        # Call LLM
        code = self._call_llm(system_prompt, user_prompt, temperature=0.1)
        if not code:
            return ""
        
        code = self._extract_code(code)
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(code)
        
        logger.info(f"Optimized code saved: {output_path}")
        return output_path
    
    def _load_prompt(self, prompt_file: str) -> str:
        """Load a prompt file."""
        prompt_path = Path(self.config.prompts_dir) / prompt_file
        if prompt_path.exists():
            with open(prompt_path) as f:
                return f.read()
        
        # Fallback to default
        logger.warning(f"Prompt not found: {prompt_path}, using default")
        default_path = Path(self.config.prompts_dir) / "triton_high_correctness.txt"
        if default_path.exists():
            with open(default_path) as f:
                return f.read()
        
        return "Generate efficient Triton kernel code for the given PyTorch implementation."
    
    def _build_user_prompt(self, problem_code: str, problem_type: str) -> str:
        """Build user prompt for generation."""
        return f"""Convert the following PyTorch code to an optimized Triton kernel.

PROBLEM TYPE: {problem_type}

PYTORCH CODE:
```python
{problem_code}
```

REQUIREMENTS:
1. Create a `ModelNew` class that matches the interface of `Model`
2. Use Triton kernels with @triton.autotune for auto-tuning
3. Ensure numerical accuracy (max diff < 0.1 from reference)
4. Optimize for MI350/gfx950 architecture
5. Use fp32 for intermediate calculations when needed for stability

Return ONLY the complete Python code with ModelNew class."""
    
    def _build_optimization_prompt(
        self,
        problem_code: str,
        prev_code: str,
        speedup: float,
        profiler_feedback: str,
        problem_type: str,
    ) -> str:
        """Build prompt for optimization."""
        prompt = f"""OPTIMIZE the following Triton kernel for better performance.

PROBLEM TYPE: {problem_type}
CURRENT SPEEDUP: {speedup:.2f}x (target: >= 1.0x)

ORIGINAL PYTORCH CODE:
```python
{problem_code}
```

CURRENT TRITON CODE TO OPTIMIZE:
```python
{prev_code}
```

"""
        if profiler_feedback:
            prompt += f"""
PROFILER FEEDBACK:
{profiler_feedback}

"""
        
        prompt += """CRITICAL RULES:
1. When using @triton.autotune, do NOT manually pass BLOCK_SIZE in kernel call
2. Do NOT use tl.tanh() - it doesn't exist on AMD. Use: x * (27 + x*x) / (27 + 9*x*x)
3. Do NOT use tl.extra.cuda.libdevice.* functions - not supported on AMD
4. Keep fp32 intermediate calculations for numerical stability
5. Use fast approximations: GELU ≈ x * tl.sigmoid(1.702 * x)

Return ONLY the complete optimized Python code with ModelNew class."""
        
        return prompt
    
    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """Call the LLM API."""
        try:
            response = self.client.messages.create(
                model=self.config.llm_model,
                max_tokens=8192,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return ""
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to extract from code blocks
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no code blocks, check if entire response is code
        if "import" in response and "class ModelNew" in response:
            return response.strip()
        
        # Try to find code starting with imports
        match = re.search(r'(import\s+.*?)(?:\Z|```)', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return response.strip()

