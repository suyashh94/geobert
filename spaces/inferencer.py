"""Inference utilities for GeoBERT model with HuggingFace Hub support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from config import ModelConfig
from huggingface_hub import hf_hub_download
from model import GeoBERTModel
from normalization import NormalizationStats
from transformers import AutoTokenizer


class Inferencer:
    """Inference class for GeoBERT geocoding model.

    Downloads model weights from HuggingFace Hub on initialization.

    :param repo_id: HuggingFace Hub repository ID (e.g., 'username/geobert-nyc').
    :param device: Device to run inference on. If None, auto-detects GPU/CPU.
    :param cache_dir: Optional cache directory for downloaded files.

    Example::

        inferencer = Inferencer("username/geobert-nyc")
        lat, lon = inferencer.predict("123 Main Street, Manhattan, NY 10001")
        print(f"Coordinates: {lat[0]:.6f}, {lon[0]:.6f}")
    """

    def __init__(
        self,
        repo_id: str = "YOUR_HF_USERNAME/geobert-nyc",
        device: torch.device | str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.repo_id = repo_id

        # Load configuration
        self.config = ModelConfig()

        # Set device (CPU-only for free tier Spaces)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        # Download and load normalization stats from Hub
        print(f"Downloading normalization stats from {repo_id}...")
        norm_stats_path = hf_hub_download(
            repo_id=repo_id,
            filename="norm_stats.json",
            cache_dir=cache_dir,
        )
        self.norm_stats = NormalizationStats.load(Path(norm_stats_path))

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_model_name)
        self.max_seq_length = 32

        # Download and load model from Hub
        self.model = self._load_model(cache_dir)

    def _load_model(self, cache_dir: str | None = None) -> GeoBERTModel:
        """Download and load model from HuggingFace Hub.

        :param cache_dir: Optional cache directory.
        :return: Loaded GeoBERTModel in eval mode.
        """
        # Download model checkpoint
        print(f"Downloading model from {self.repo_id}...")
        checkpoint_path = hf_hub_download(
            repo_id=self.repo_id,
            filename="best_model.pt",
            cache_dir=cache_dir,
        )

        # Create model and load state dict
        model = GeoBERTModel(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        print(f"Model loaded on device: {self.device}")
        return model

    def predict(
        self,
        addresses: str | list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict geographic coordinates for one or more addresses.

        :param addresses: Single address string or list of addresses.
        :return: Tuple of (latitudes, longitudes) as numpy arrays.
        """
        # Handle single address
        if isinstance(addresses, str):
            addresses = [addresses]

        # Tokenize
        encoding = self.tokenizer(
            addresses,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Forward pass
        with torch.no_grad():
            predictions = self.model(input_ids, attention_mask)

        # Denormalize predictions
        lat, lon = self.norm_stats.denormalize(predictions.cpu())

        return lat.numpy(), lon.numpy()

    def predict_batch(
        self,
        addresses: list[str],
        batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict coordinates for a large list of addresses in batches.

        More memory-efficient than predict() for large datasets.

        :param addresses: List of address strings.
        :param batch_size: Number of addresses to process at once.
        :return: Tuple of (latitudes, longitudes) as numpy arrays.
        """
        all_lats = []
        all_lons = []

        for i in range(0, len(addresses), batch_size):
            batch = addresses[i : i + batch_size]
            lats, lons = self.predict(batch)
            all_lats.append(lats)
            all_lons.append(lons)

        return np.concatenate(all_lats), np.concatenate(all_lons)
