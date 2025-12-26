"""Inference utilities for GeoBERT model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from geobert.config import ModelConfig
from geobert.device import get_device
from geobert.model import GeoBERTModel
from geobert.normalization import NormalizationStats


class Inferencer:
    """Inference class for GeoBERT geocoding model.

    Loads a trained model checkpoint and provides methods for predicting
    geographic coordinates from addresses.

    :param checkpoint_dir: Directory containing model checkpoint and normalization stats.
    :param device: Device to run inference on. If None, auto-detects GPU/CPU.

    Example::

        inferencer = Inferencer("outputs/checkpoints")
        lat, lon = inferencer.predict("123 Main Street, Manhattan, NY 10001")
        print(f"Coordinates: {lat[0]:.6f}, {lon[0]:.6f}")
    """

    def __init__(
        self,
        checkpoint_dir: Path | str = "outputs/checkpoints",
        device: torch.device | str | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)

        # Load configuration
        self.config = ModelConfig()

        # Set device
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        # Load normalization stats
        norm_stats_path = self.checkpoint_dir / "norm_stats.json"
        if not norm_stats_path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {norm_stats_path}")
        self.norm_stats = NormalizationStats.load(norm_stats_path)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_model_name)
        self.max_seq_length = 32  # Default from DataConfig

        # Load model
        self.model = self._load_model()

    def _load_model(self) -> GeoBERTModel:
        """Load model from checkpoint.

        :return: Loaded GeoBERTModel in eval mode.
        """
        # Try best_model.pt first, then checkpoint_epoch.pt
        checkpoint_path = self.checkpoint_dir / "best_model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = self.checkpoint_dir / "checkpoint_epoch.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found in {self.checkpoint_dir}. "
                "Expected 'best_model.pt' or 'checkpoint_epoch.pt'."
            )

        # Create model and load state dict
        model = GeoBERTModel(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

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
