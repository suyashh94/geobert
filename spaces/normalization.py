"""Z-score normalization for geographic coordinates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


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

    def denormalize(self, predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert normalized predictions back to lat/long.

        :param predictions: Tensor of shape (N, 2) with normalized [lat, lon].
        :return: Tuple of (latitude, longitude) tensors.
        """
        lat = predictions[:, 0] * self.lat_std + self.lat_mean
        lon = predictions[:, 1] * self.lon_std + self.lon_mean
        return lat, lon

    def denormalize_sigma(
        self, sigma_lat: torch.Tensor, sigma_lon: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denormalize sigma (std dev) values back to degrees.

        Unlike denormalize(), sigma values only need scaling (no mean offset)
        since they represent spreads, not absolute positions.

        :param sigma_lat: Tensor of normalized latitude std devs.
        :param sigma_lon: Tensor of normalized longitude std devs.
        :return: Tuple of (sigma_lat, sigma_lon) in degrees.
        """
        return sigma_lat * self.lat_std, sigma_lon * self.lon_std

    @classmethod
    def load(cls, path: Path | str) -> NormalizationStats:
        """Load normalization stats from JSON file.

        :param path: Path to the JSON file.
        :return: NormalizationStats instance.
        """
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
