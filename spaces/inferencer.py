"""Inference utilities for GeoBERT model with HuggingFace Hub support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from config import ModelConfig
from helper import sample_mdn
from huggingface_hub import hf_hub_download
from model import GeoBERTMDNModel, GeoBERTModel
from normalization import NormalizationStats
from transformers import AutoTokenizer


class Inferencer:
    """Inference class for GeoBERT geocoding model.

    Downloads model weights from HuggingFace Hub on initialization,
    or loads from a local directory if specified.

    :param repo_id: HuggingFace Hub repository ID (e.g., 'username/geobert-nyc').
    :param local_dir: Optional local directory containing model files (for testing).
    :param device: Device to run inference on. If None, auto-detects GPU/CPU.
    :param cache_dir: Optional cache directory for downloaded files.
    :param model_type: Model type: 'regression' or 'mdn'.

    Example::

        # Regression model from HuggingFace Hub
        inferencer = Inferencer("suyash94/geobert-nyc")

        # MDN model from local directory
        inferencer = Inferencer(local_dir="../outputs/checkpoints", model_type="mdn")

        lat, lon = inferencer.predict("123 Main Street, Manhattan, NY 10001")
        print(f"Coordinates: {lat[0]:.6f}, {lon[0]:.6f}")
    """

    def __init__(
        self,
        repo_id: str = "suyash94/geobert-nyc",
        local_dir: str | Path | None = None,
        device: torch.device | str | None = None,
        cache_dir: str | None = None,
        model_type: str = "regression",
    ) -> None:
        self.repo_id = repo_id
        self.local_dir = Path(local_dir) if local_dir else None
        self.model_type = model_type

        # Load configuration
        self.config = ModelConfig()

        # Set device (CPU-only for free tier Spaces)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        # Load normalization stats (local or from Hub)
        if self.local_dir:
            print(f"Loading normalization stats from local: {self.local_dir}")
            norm_stats_path = self.local_dir / "norm_stats.json"
        else:
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

        # Load model (local or from Hub)
        self.model = self._load_model(cache_dir)

    def _load_model(self, cache_dir: str | None = None) -> GeoBERTModel | GeoBERTMDNModel:
        """Load model from local directory or HuggingFace Hub.

        :param cache_dir: Optional cache directory.
        :return: Loaded model in eval mode.
        """
        # Determine checkpoint filename based on model type
        if self.model_type == "mdn":
            checkpoint_filename = "best_model_mdn.pt"
        else:
            checkpoint_filename = "best_model.pt"

        # Get checkpoint path (local or from Hub)
        if self.local_dir:
            print(f"Loading {self.model_type} model from local: {self.local_dir}")
            checkpoint_path = self.local_dir / checkpoint_filename
            if not checkpoint_path.exists():
                # Fallback to generic checkpoint
                checkpoint_path = self.local_dir / "checkpoint_epoch.pt"
        else:
            print(f"Downloading {self.model_type} model from {self.repo_id}...")
            checkpoint_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=checkpoint_filename,
                cache_dir=cache_dir,
            )

        # Create model and load state dict
        if self.model_type == "mdn":
            model = GeoBERTMDNModel(self.config, self.config.mdn_num_mixtures)
        else:
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
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict geographic coordinates for one or more addresses.

        :param addresses: Single address string or list of addresses.
        :param deterministic: If True and model is MDN, use highest-weight component mean.
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
            if self.model_type == "regression":
                predictions = self.model(input_ids, attention_mask)
            elif self.model_type == "mdn":
                pi_logits, mu_lat, mu_lon, sigma_lat, sigma_lon = self.model(
                    input_ids, attention_mask
                )
                sampled_lat, sampled_lon = sample_mdn(
                    pi_logits,
                    mu_lat,
                    mu_lon,
                    sigma_lat,
                    sigma_lon,
                    temperature=1.0,
                    sample_max_mean=deterministic,
                )
                predictions = torch.stack([sampled_lat, sampled_lon], dim=1)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

        # Denormalize predictions
        lat, lon = self.norm_stats.denormalize(predictions.cpu())

        return lat.numpy(), lon.numpy()

    def predict_mdn_raw(
        self,
        addresses: str | list[str],
    ) -> dict[str, np.ndarray]:
        """Predict with full MDN outputs for confidence visualization.

        Returns deterministic predictions (highest-weight component mean) along with
        uncertainty estimates (sigma) for confidence interval visualization.

        :param addresses: Single address string or list of addresses.
        :return: Dictionary with keys:
            - 'lat', 'lon': deterministic predictions (highest-weight component mean)
            - 'sigma_lat', 'sigma_lon': denormalized std devs from highest-weight component (degrees)
            - 'pi': mixture weights after softmax
        :raises ValueError: If model_type is not 'mdn'.
        """
        if self.model_type != "mdn":
            raise ValueError("predict_mdn_raw() requires model_type='mdn'")

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
            pi_logits, mu_lat, mu_lon, sigma_lat, sigma_lon = self.model(
                input_ids, attention_mask
            )

            # Get deterministic predictions with sigma using sample_mdn
            pred_lat, pred_lon, sel_sigma_lat, sel_sigma_lon = sample_mdn(
                pi_logits,
                mu_lat,
                mu_lon,
                sigma_lat,
                sigma_lon,
                sample_max_mean=True,
                return_sigma=True,
            )

            # Get mixture weights
            pi = torch.softmax(pi_logits, dim=-1)

            # Stack predictions for denormalization
            predictions = torch.stack([pred_lat, pred_lon], dim=1)

        # Denormalize predictions and sigmas
        lat, lon = self.norm_stats.denormalize(predictions.cpu())
        sigma_lat_denorm, sigma_lon_denorm = self.norm_stats.denormalize_sigma(
            sel_sigma_lat.cpu(), sel_sigma_lon.cpu()
        )

        return {
            "lat": lat.numpy(),
            "lon": lon.numpy(),
            "sigma_lat": sigma_lat_denorm.numpy(),
            "sigma_lon": sigma_lon_denorm.numpy(),
            "pi": pi.cpu().numpy(),
        }

    def predict_batch(
        self,
        addresses: list[str],
        batch_size: int = 64,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict coordinates for a large list of addresses in batches.

        More memory-efficient than predict() for large datasets.

        :param addresses: List of address strings.
        :param batch_size: Number of addresses to process at once.
        :param deterministic: If True and model is MDN, use highest-weight component mean.
        :return: Tuple of (latitudes, longitudes) as numpy arrays.
        """
        all_lats = []
        all_lons = []

        for i in range(0, len(addresses), batch_size):
            batch = addresses[i : i + batch_size]
            lats, lons = self.predict(batch, deterministic=deterministic)
            all_lats.append(lats)
            all_lons.append(lons)

        return np.concatenate(all_lats), np.concatenate(all_lons)
