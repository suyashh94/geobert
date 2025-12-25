# GeoBERT

Deep learning project using PyTorch with automatic GPU/CPU fallback.

## Development Setup

This project uses a dev container with:
- **Python 3.12**
- **PyTorch 2.7+** with CUDA 12.8 support
- **uv** for fast dependency management

### Prerequisites

- Docker Desktop with WSL2 backend (Windows) or Docker Engine (Linux/Mac)
- VS Code with the Dev Containers extension
- For GPU support: NVIDIA GPU with drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Getting Started

1. **Open in VS Code**:
   ```bash
   code geobert/
   ```

2. **Open in Dev Container**:
   - Press `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"
   - Or click the green button in the bottom-left corner

3. **Wait for setup**:
   - The container will build and install dependencies automatically
   - GPU availability is detected and PyTorch is configured accordingly

### Manual Setup (without Dev Container)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.12

# Install with GPU support
uv sync --extra cuda

# Or install CPU-only
uv sync --extra cpu

# Activate environment
source .venv/bin/activate
```

### Verify Installation

```bash
# Check PyTorch and GPU status
python -m geobert.device
```

## Project Structure

```
geobert/
├── .devcontainer/
│   ├── devcontainer.json      # GPU-enabled config (default)
│   ├── devcontainer-cpu.json  # CPU-only config
│   ├── Dockerfile
│   └── post-create.sh
├── src/
│   └── geobert/
│       ├── __init__.py
│       └── device.py          # GPU/CPU device utilities
├── tests/
├── pyproject.toml             # Project config with uv
└── README.md
```

## GPU vs CPU Mode

The dev container automatically detects GPU availability:

| Environment | What happens |
|-------------|--------------|
| With NVIDIA GPU | Uses CUDA 12.8, full GPU acceleration |
| Without GPU | Falls back to CPU-only PyTorch |

To explicitly use CPU-only mode, rename `devcontainer-cpu.json` to `devcontainer.json`.

## Dependencies

Core ML stack:
- `torch` 2.7+ - Deep learning framework
- `transformers` - Hugging Face transformers
- `datasets` - Hugging Face datasets
- `accelerate` - Distributed training
- `wandb` - Experiment tracking

Development:
- `ruff` - Linting and formatting
- `pytest` - Testing
- `jupyter` - Notebooks
