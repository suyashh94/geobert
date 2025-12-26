# GeoBERT - NYC Address Geocoding with BERT

A deep learning model that predicts geographic coordinates (latitude/longitude) from New York City addresses. The model uses a fine-tuned BERT architecture to learn the relationship between address text and location.

## Try It Now

**[Launch the Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/suyash94/geobert-nyc-geocoder)**

Enter any NYC address and get predicted coordinates with an interactive map.

## What This Project Does

Traditional geocoding relies on address parsing and database lookups. GeoBERT takes a different approach: it treats geocoding as a **text-to-coordinates regression problem**, using a transformer model to learn spatial patterns directly from address text.

**Input:** `350 5th Avenue, Manhattan, NY 10118`
**Output:** `(40.748817, -73.985428)`

### Model Architecture

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
# Single GPU
geobert-train

# Multi-GPU with DDP
torchrun --nproc_per_node=4 -m geobert.cli

# Debug mode (1000 samples, quick iteration)
geobert-train --debug
```

Training logs are tracked with MLflow. View them with:
```bash
mlflow ui
```

### 4. Evaluate

Open `notebooks/04_evaluation.ipynb` to:
- Run inference on the test set
- View metrics (MSE, haversine distance)
- Explore predictions on interactive Folium maps

### 5. Use the Model

```python
from geobert import Inferencer

# Load trained model
inferencer = Inferencer("outputs/checkpoints")

# Predict coordinates
lat, lon = inferencer.predict("123 Main Street, Manhattan, NY 10001")
print(f"Coordinates: {lat[0]:.6f}, {lon[0]:.6f}")
```

## Project Structure

```
geobert/
├── src/geobert/           # Main package
│   ├── cli.py             # Training CLI (geobert-train)
│   ├── model.py           # GeoBERTModel architecture
│   ├── trainer.py         # Training loop with DDP support
│   ├── dataset.py         # Data loading and preprocessing
│   ├── inferencer.py      # Inference wrapper
│   └── metrics.py         # Haversine distance, MSE, etc.
├── src/data/
│   └── fetch_nyc_data.py  # NYC Open Data fetcher
├── notebooks/
│   ├── 01_eda.ipynb                   # Data exploration
│   ├── 02_map_visualizations.ipynb    # Folium maps
│   ├── 03_model_design_decisions.ipynb # Architecture rationale
│   └── 04_evaluation.ipynb            # Model evaluation
├── spaces/                # HuggingFace Spaces deployment
│   ├── app.py             # Gradio interface
│   └── ...
└── outputs/
    └── checkpoints/       # Saved models
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
# Check GPU/device status
python -m geobert.device

# Run tests
pytest

# Lint
ruff check src tests

# Format
ruff format src tests
```

## Resources

- **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/suyash94/geobert-nyc-geocoder)
- **Model Weights:** [HuggingFace Hub](https://huggingface.co/suyash94/geobert-nyc)
- **Training Data:** [NYC Open Data - Address Points](https://data.cityofnewyork.us/)

## License

MIT
