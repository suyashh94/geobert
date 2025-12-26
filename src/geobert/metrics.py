"""Evaluation metrics for geocoding model."""

from dataclasses import dataclass

import torch

from geobert.normalization import NormalizationStats

# Earth radius in meters
EARTH_RADIUS_M = 6_371_000


@dataclass
class GeoMetrics:
    """Geocoding evaluation metrics.

    :param mse: Mean squared error on normalized coordinates.
    :param mae_lat: Mean absolute error for latitude (degrees).
    :param mae_lon: Mean absolute error for longitude (degrees).
    :param rmse_lat: Root mean squared error for latitude (degrees).
    :param rmse_lon: Root mean squared error for longitude (degrees).
    :param mean_distance_m: Mean haversine distance error in meters.
    :param median_distance_m: Median haversine distance error in meters.
    """

    mse: float
    mae_lat: float
    mae_lon: float
    rmse_lat: float
    rmse_lon: float
    mean_distance_m: float
    median_distance_m: float


def haversine_distance_m(
    lat1: torch.Tensor,
    lon1: torch.Tensor,
    lat2: torch.Tensor,
    lon2: torch.Tensor,
) -> torch.Tensor:
    """Compute haversine distance in meters.

    :param lat1: Predicted latitudes in degrees.
    :param lon1: Predicted longitudes in degrees.
    :param lat2: True latitudes in degrees.
    :param lon2: True longitudes in degrees.
    :return: Distances in meters.
    """
    lat1_rad = torch.deg2rad(lat1)
    lat2_rad = torch.deg2rad(lat2)
    delta_lat = torch.deg2rad(lat2 - lat1)
    delta_lon = torch.deg2rad(lon2 - lon1)

    a = (
        torch.sin(delta_lat / 2) ** 2
        + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(delta_lon / 2) ** 2
    )
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return EARTH_RADIUS_M * c


def compute_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    norm_stats: NormalizationStats,
) -> GeoMetrics:
    """Compute geocoding metrics from normalized predictions.

    :param predictions: Model predictions (normalized) of shape (N, 2).
    :param labels: True labels (normalized) of shape (N, 2).
    :param norm_stats: Normalization statistics for denormalization.
    :return: GeoMetrics with all evaluation metrics.
    """
    # MSE on normalized values (training loss)
    mse = torch.mean((predictions - labels) ** 2).item()

    # Denormalize to actual coordinates
    pred_lat, pred_lon = norm_stats.denormalize(predictions)
    true_lat, true_lon = norm_stats.denormalize(labels)

    # MAE in degrees
    mae_lat = torch.mean(torch.abs(pred_lat - true_lat)).item()
    mae_lon = torch.mean(torch.abs(pred_lon - true_lon)).item()

    # RMSE in degrees
    rmse_lat = torch.sqrt(torch.mean((pred_lat - true_lat) ** 2)).item()
    rmse_lon = torch.sqrt(torch.mean((pred_lon - true_lon) ** 2)).item()

    # Distance error in meters
    distances = haversine_distance_m(pred_lat, pred_lon, true_lat, true_lon)
    mean_distance_m = distances.mean().item()
    median_distance_m = distances.median().item()

    return GeoMetrics(
        mse=mse,
        mae_lat=mae_lat,
        mae_lon=mae_lon,
        rmse_lat=rmse_lat,
        rmse_lon=rmse_lon,
        mean_distance_m=mean_distance_m,
        median_distance_m=median_distance_m,
    )
