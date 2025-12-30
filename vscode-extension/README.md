# AMD HIP/Triton Generator

Generate optimized **Triton/HIP GPU kernels** from PyTorch code for AMD Instinct GPUs using LLM.

## Features

- **One-click Generation** — Select PyTorch code, right-click, and generate optimized Triton kernels
- **Auto Evaluation** — Automatically validates correctness and measures speedup
- **Multi-attempt Retry** — Retries with error feedback if compilation or tests fail
- **Triton Optimization** — Optimize existing Triton kernels for better performance

## Quick Start

### 1. Configure API Key

1. Open **Settings** (`Ctrl+,`)
2. Search for `hipGenerator.amdApiKey`
3. Enter your AMD LLM Gateway API Key (get one at [llm.amd.com](https://llm.amd.com))

### 2. Generate a Kernel

1. Open a Python file with PyTorch code
2. **Select** the code you want to convert (a `Model` class with `forward` method)
3. **Right-click** → Select **"Generate Triton Kernel"** or **"Generate HIP Kernel"**
4. Wait for generation and evaluation
5. If successful, the generated code is inserted at the end of your file

### 3. Optimize Existing Triton Code

1. Select existing Triton kernel code in the editor
2. **Right-click** → **"Optimize Triton Kernel"**
3. The optimized code will be appended to your file

## Input Code Format

Your PyTorch code should include:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        return torch.matmul(x, y)

def get_inputs():
    """Returns test input tensors"""
    return [
        torch.randn(1024, 1024, dtype=torch.float16, device='cuda'),
        torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
    ]

def get_init_inputs():
    """Returns Model.__init__ arguments"""
    return []
```

## Understanding Results

| Status | Meaning |
|--------|---------|
| Compile ✓ | Code compiles without errors |
| Accuracy ✓ | Output matches PyTorch reference |
| Speedup 2.5x | 2.5 times faster than PyTorch |

## Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `amdApiKey` | (required) | AMD LLM Gateway API Key |
| `maxAttempts` | 3 | Max retry attempts on failure |
| `targetSpeedup` | 1.0 | Target speedup threshold |
| `temperature` | 0.3 | LLM temperature (0-1) |
| `pythonPath` | python3 | Python interpreter path |

## Sidebar Panel

The sidebar shows:
- **Current Task** — Running generation with cancel option
- **Generation History** — Past 20 generations with results
- Click **"View Code"** to see generated code

## Requirements

- **Python 3.8+** with PyTorch and Triton installed
- **AMD GPU** (Instinct MI series recommended)
- **ROCm** runtime for GPU execution

## Troubleshooting

### "AMD API Key not configured"
Set your API key in Settings → `hipGenerator.amdApiKey`

### "HipGenerator not found"
The extension includes the Python backend. Ensure Python dependencies are installed:
```bash
pip install torch triton
```

### Generation fails repeatedly
- Increase `maxAttempts` to 5-10
- Try a simpler operation first
- Check the Output panel ("HIP Generator") for detailed logs

## License

MIT License - AMD © 2025
