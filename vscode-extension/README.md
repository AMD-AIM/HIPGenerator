# HIP/Triton Generator - VSCode Extension

Generate optimized Triton/HIP GPU kernels from PyTorch code using LLM.

## Features

- **Right-click Generate**: Select PyTorch code, right-click, and generate Triton or HIP kernels
- **Auto-infer Inputs**: Automatically infers `get_inputs()` and `get_init_inputs()` from your code
- **Interactive Panel**: Modify inferred inputs and regenerate code
- **Evaluation**: Test generated kernels for correctness and performance
- **AMD GPU Optimized**: Uses AMD LLM Gateway with optimizations for MI350 GPUs

## Quick Start

### 1. Install Dependencies

```bash
cd /root/HipGenerator/vscode-extension
npm install
npm run compile
```

### 2. Configure API Key

1. Open VSCode Settings (Ctrl+,)
2. Search for "HIP Generator"
3. Set your AMD API Key in `hipGenerator.amdApiKey`

Or use the command line:
```bash
export LLM_GATEWAY_KEY="your_api_key_here"
```

### 3. Use the Extension

1. Open a Python file with PyTorch code
2. Select the code you want to convert (should include a `Model` class)
3. Right-click and select "Generate Triton Kernel" or "Generate HIP Kernel"
4. The generator panel opens with:
   - Auto-inferred `get_inputs()` and `get_init_inputs()`
   - Options to modify inputs
   - Generate, Evaluate, and Save buttons

## Requirements

This extension is designed to run in a Docker container with:

- **ROCm**: AMD GPU drivers and runtime
- **PyTorch**: With ROCm support
- **Triton**: For Triton backend
- **Python 3.8+**: With required packages

The extension assumes these dependencies are already installed in the container.

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `hipGenerator.amdApiKey` | "" | AMD LLM Gateway API Key |
| `hipGenerator.defaultBackend` | "triton" | Default backend (triton/hip) |
| `hipGenerator.maxAttempts` | 3 | Max generation attempts |
| `hipGenerator.temperature` | 0.3 | LLM temperature |
| `hipGenerator.targetSpeedup` | 1.0 | Target speedup vs baseline |
| `hipGenerator.pythonPath` | "python3" | Python interpreter path |

## Input Code Format

Your PyTorch code should follow this structure:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        return self.linear(x)

# Optional - will be auto-inferred if not provided
def get_inputs():
    return [torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')]

def get_init_inputs():
    return [1024]  # hidden_dim
```

## Generated Code Format

The extension generates code with a `ModelNew` class that replaces PyTorch operations with Triton/HIP kernels:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(...):
    # Optimized Triton kernel
    ...

class ModelNew(nn.Module):
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
    
    def forward(self, x):
        # Uses Triton kernel instead of torch.mm
        return triton_matmul(x, self.weight)
```

## Development

### Build

```bash
npm install
npm run compile
```

### Watch Mode

```bash
npm run watch
```

### Debug

1. Open the extension folder in VSCode
2. Press F5 to launch a new Extension Development Host
3. Test the extension in the new window

### Package

```bash
npm run package
# Creates hip-triton-generator-0.1.0.vsix
```

## Architecture

```
vscode-extension/
├── src/
│   ├── extension.ts          # Extension entry point
│   ├── panels/
│   │   └── GeneratorPanel.ts # Main WebView panel
│   └── services/
│       ├── PythonBackend.ts  # Python process management
│       └── CodeAnalyzer.ts   # Input inference logic
├── resources/
│   └── icon.svg              # Extension icon
└── package.json              # Extension manifest
```

## Troubleshooting

### "AMD API Key not configured"
Set `hipGenerator.amdApiKey` in VSCode settings or export `LLM_GATEWAY_KEY`.

### "HipGenerator not found"
Ensure the extension is in the same directory as the HipGenerator Python code, or that `/root/HipGenerator` exists.

### "Generation failed"
Check the "HIP Generator" output channel in VSCode for detailed error messages.

### Slow generation
Generation can take 30-60 seconds depending on code complexity. The extension shows a spinner during generation.

## License

MIT










