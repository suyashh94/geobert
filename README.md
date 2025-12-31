# GeoBERT - NYC Address Geocoding with BERT

A deep learning model that predicts geographic coordinates (latitude/longitude) from New York City addresses. The model uses a fine-tuned BERT architecture to learn the relationship between address text and location.

## Try It Now

**[Launch the Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/suyash94/geobert-nyc-geocoder)**

Enter any NYC address and get predicted coordinates with an interactive map.

## What This Project Does

Traditional geocoding relies on address parsing and database lookups. GeoBERT takes a different approach: it treats geocoding as a **text-to-coordinates regression problem**, using a transformer model to learn spatial patterns directly from address text.

**Input:** `350 5th Avenue, Manhattan, NY 10118`
**Output:** `(40.748817, -73.985428)`

### Model Architectures

GeoBERT supports two model variants:

#### 1. Regression Model (Default)

```
Address Text
    ↓
BERT Tokenizer (max 32 tokens)
    ↓
Tiny BERT (google/bert_uncased_L-2_H-128_A-2)
    - 2 transformer layers
    - 128 hidden dimensions
    - 4.4M parameters
    ↓
[CLS] Token Embedding (128-dim)
    ↓
Regression Head: Linear(256) → ReLU → Linear(2)
    ↓
[latitude, longitude] (z-score normalized)
```

#### 2. MDN Model (Mixture Density Network)

For probabilistic predictions with uncertainty estimates:

```
Address Text
    ↓
Tiny BERT → [CLS] Embedding (128-dim) → ReLU
    ↓
Three parallel heads:
    ├── π head: Linear(K) → Softmax     [mixture weights]
    ├── μ head: Linear(K×2)             [means for lat/lon]
    └── σ head: Linear(K×2) → Softplus  [std devs for lat/lon]
    ↓
K Gaussian mixture components for probabilistic output
```

The MDN model outputs a mixture of Gaussians, enabling:
- **Uncertainty quantification** (confidence intervals)
- **Multimodal predictions** (ambiguous addresses)
- **Calibrated probability estimates**

### Training Data

- **~1M address points** from [NYC Open Data](https://data.cityofnewyork.us/)
- Covers all 5 boroughs: Manhattan, Brooklyn, Queens, Bronx, Staten Island
- Split: 80% train, 10% validation, 10% test

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/suyash94/geobert.git
cd geobert

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Fetch Training Data

```bash
python -m src.data.fetch_nyc_data
```

Downloads ~968K address records from NYC Open Data API.

### 3. Train the Model

```bash
# Single GPU - Regression model (default)
geobert-train

# Single GPU - MDN model
geobert-train --training-mode mdn --mdn-num-mixtures 5

# Multi-GPU with DDP
torchrun --nproc_per_node=4 -m geobert.cli

# Debug mode (1000 samples, quick iteration)
geobert-train --debug
```

#### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--training-mode` | `regression` | Model type: `regression` or `mdn` |
| `--mdn-num-mixtures` | `5` | Number of Gaussian mixtures (MDN only) |
| `--batch-size` | `256` | Batch size per GPU |
| `--epochs` | `10` | Number of training epochs |
| `--learning-rate` | `2e-5` | AdamW learning rate |
| `--debug` | - | Quick test: 1000 samples, 2 epochs |

Training logs are tracked with MLflow:
```bash
mlflow ui
```

### 4. Evaluate

Open the evaluation notebooks:

- `notebooks/04_evaluation.ipynb` - Metrics, error analysis, prediction maps
- `notebooks/05_mdn_confidence_visualization.ipynb` - MDN uncertainty visualization

### 5. Use the Model

#### Regression Model

```python
from geobert import Inferencer

# Load trained model
inferencer = Inferencer("outputs/checkpoints")

# Predict coordinates
lat, lon = inferencer.predict("123 Main Street, Manhattan, NY 10001")
print(f"Coordinates: {lat[0]:.6f}, {lon[0]:.6f}")
```

#### MDN Model

```python
from geobert import Inferencer

# Load MDN model
inferencer = Inferencer("outputs/checkpoints", model_type="mdn")

# Deterministic prediction (highest-weight component mean)
lat, lon = inferencer.predict("123 Main Street, Manhattan, NY", deterministic=True)

# Get full MDN outputs with uncertainty
results = inferencer.predict_mdn_raw("123 Main Street, Manhattan, NY")
print(f"Prediction: ({results['lat'][0]:.6f}, {results['lon'][0]:.6f})")
print(f"Uncertainty: σ_lat={results['sigma_lat'][0]:.6f}°, σ_lon={results['sigma_lon'][0]:.6f}°")
```

## Project Structure

```
geobert/
├── src/geobert/              # Main package
│   ├── cli.py                # Training CLI (geobert-train)
│   ├── config.py             # Configuration dataclasses
│   ├── model.py              # GeoBERTModel & GeoBERTMDNModel
│   ├── trainer.py            # Training loop with DDP support
│   ├── dataset.py            # Data loading and preprocessing
│   ├── inferencer.py         # Inference wrapper
│   ├── loss.py               # MDNLoss function
│   ├── helper.py             # MDN sampling utilities
│   ├── metrics.py            # Haversine distance, MSE, etc.
│   ├── normalization.py      # Z-score normalization
│   └── device.py             # GPU/CPU detection
├── src/data/
│   └── fetch_nyc_data.py     # NYC Open Data fetcher
├── notebooks/
│   ├── 01_eda.ipynb                        # Data exploration
│   ├── 02_map_visualizations.ipynb         # Folium maps
│   ├── 03_model_design_decisions.ipynb     # Architecture rationale
│   ├── 04_evaluation.ipynb                 # Model evaluation (regression & MDN)
│   └── 05_mdn_confidence_visualization.ipynb  # MDN uncertainty visualization
├── spaces/                   # HuggingFace Spaces deployment
│   ├── app.py                # Gradio interface
│   └── ...
└── outputs/
    └── checkpoints/          # Saved models
```

## Development

### Prerequisites

- Python 3.12+
- CUDA 12.x (optional, for GPU training)

### Dev Container (Recommended)

This project includes a VS Code dev container with GPU support:

1. Install [Docker](https://docs.docker.com/get-docker/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Open in VS Code → "Reopen in Container"

### Commands

```bash
# Install dev dependencies
uv sync --group dev

# Check GPU/device status
python -m geobert.device

# Run tests
pytest

# Lint
ruff check src tests

# Format
ruff format src tests

# Type check
mypy src
```

## Resources

- **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/suyash94/geobert-nyc-geocoder)
- **Model Weights:** [HuggingFace Hub](https://huggingface.co/suyash94/geobert-nyc)
- **Training Data:** [NYC Open Data - Address Points](https://data.cityofnewyork.us/)

## License

MIT
