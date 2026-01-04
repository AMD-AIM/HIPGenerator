"""
Configuration management for HIPGenerator.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class Config:
    """Configuration for HIPGenerator."""
    
    # API settings
    llm_gateway_key: str = field(default_factory=lambda: os.environ.get('LLM_GATEWAY_KEY', ''))
    llm_base_url: str = "https://llm-api.amd.com/Anthropic"
    llm_model: str = "claude-opus-4"
    
    # Generation settings
    backend: str = "triton"
    temperature: float = 0.1
    num_samples: int = 1
    
    # Optimization settings
    max_optimize_rounds: int = 2
    target_speedup: float = 1.0
    
    # Paths
    output_dir: str = "results"
    prompts_dir: str = "prompts"
    datasets_dir: str = "datasets"
    
    # Profiler settings
    enable_profiler: bool = True
    profiler_timeout: int = 300
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables."""
        return cls(
            llm_gateway_key=os.environ.get('LLM_GATEWAY_KEY', ''),
            output_dir=os.environ.get('HIPGEN_OUTPUT_DIR', 'results'),
            log_level=os.environ.get('HIPGEN_LOG_LEVEL', 'INFO'),
        )
    
    def validate(self) -> List[str]:
        """Validate configuration. Returns list of errors."""
        errors = []
        if not self.llm_gateway_key:
            errors.append("LLM_GATEWAY_KEY not set")
        if self.backend not in ['triton', 'hip']:
            errors.append(f"Invalid backend: {self.backend}")
        if self.temperature < 0 or self.temperature > 2:
            errors.append(f"Invalid temperature: {self.temperature}")
        return errors


# Global default config
_default_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration."""
    global _default_config
    if _default_config is None:
        _default_config = Config.from_env()
    return _default_config


def set_config(config: Config) -> None:
    """Set the global configuration."""
    global _default_config
    _default_config = config

