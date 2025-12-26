"""Z-score normalization for geographic coordinates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class NormalizationStats:
    """Statistics for Z-score normalization of coordinates.

    :param lat_mean: Mean latitude value.
    :param lat_std: Standard deviation of latitude.
    :param lon_mean: Mean longitude value.
    :param lon_std: Standard deviation of longitude.
    """

    lat_mean: float
    lat_std: float
    lon_mean: float
    lon_std: float

    def normalize(self, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        """Normalize latitude and longitude to z-scores.

        :param lat: Tensor of latitude values.
        :param lon: Tensor of longitude values.
        :return: Tensor of shape (N, 2) with normalized [lat, lon].
        """
        lat_norm = (lat - self.lat_mean) / self.lat_std
        lon_norm = (lon - self.lon_mean) / self.lon_std
        return torch.stack([lat_norm, lon_norm], dim=-1)

    def denormalize(self, predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert normalized predictions back to lat/long.

        :param predictions: Tensor of shape (N, 2) with normalized [lat, lon].
        :return: Tuple of (latitude, longitude) tensors.
        """
        lat = predictions[:, 0] * self.lat_std + self.lat_mean
        lon = predictions[:, 1] * self.lon_std + self.lon_mean
        return lat, lon

    def save(self, path: Path) -> None:
        """Save normalization stats to JSON file.

        :param path: Path to save the JSON file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "lat_mean": self.lat_mean,
                    "lat_std": self.lat_std,
                    "lon_mean": self.lon_mean,
                    "lon_std": self.lon_std,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> NormalizationStats:
        """Load normalization stats from JSON file.

        :param path: Path to the JSON file.
        :return: NormalizationStats instance.
        """
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> NormalizationStats:
        """Compute normalization stats from a DataFrame.

        :param df: DataFrame with 'latitude' and 'longitude' columns.
        :return: NormalizationStats computed from the data.
        """
        return cls(
            lat_mean=float(df["latitude"].mean()),
            lat_std=float(df["latitude"].std()),
            lon_mean=float(df["longitude"].mean()),
            lon_std=float(df["longitude"].std()),
        )
