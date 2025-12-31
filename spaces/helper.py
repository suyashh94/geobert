"""Helper functions for MDN sampling."""

import torch
import torch.nn.functional as F


def sample_mdn(
    pi_logits: torch.Tensor,
    mu_lat: torch.Tensor,
    mu_lon: torch.Tensor,
    sigma_lat: torch.Tensor,
    sigma_lon: torch.Tensor,
    temperature: float = 1.0,
    sample_max_mean: bool = False,
    return_sigma: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """
    Sample from the mixture density network.

    :param pi_logits: Raw logits for mixture weights, shape (batch, K).
    :param mu_lat: Predicted means for latitude, shape (batch, K).
    :param mu_lon: Predicted means for longitude, shape (batch, K).
    :param sigma_lat: Predicted stddevs for latitude, shape (batch, K).
    :param sigma_lon: Predicted stddevs for longitude, shape (batch, K).
    :param temperature: Controls sampling randomness. 1.0 = normal, <1 = more deterministic, >1 = more random.
    :param sample_max_mean: If True, use mean of highest-weight component (deterministic).
    :param return_sigma: If True, also return sigma values from selected component.
    :return: Tuple of (sampled_lat, sampled_lon), each of shape (batch,).
             If return_sigma=True, returns (lat, lon, sigma_lat, sigma_lon).
    """
    batch_size = pi_logits.shape[0]
    batch_idx = torch.arange(batch_size, device=pi_logits.device)

    # Step 1: Sample which component to use for each batch element
    pi = F.softmax(pi_logits / temperature, dim=-1)  # (batch, K)

    if sample_max_mean:
        component_indices = torch.argmax(pi, dim=-1)  # (batch,)
    else:
        component_indices = torch.multinomial(pi, num_samples=1).squeeze(-1)  # (batch,)

    # Step 2: Gather the mu and sigma for the selected components
    selected_mu_lat = mu_lat[batch_idx, component_indices]  # (batch,)
    selected_mu_lon = mu_lon[batch_idx, component_indices]  # (batch,)
    selected_sigma_lat = sigma_lat[batch_idx, component_indices]  # (batch,)
    selected_sigma_lon = sigma_lon[batch_idx, component_indices]  # (batch,)

    # Step 3: Get final coordinates
    if sample_max_mean:
        # Deterministic: just use the mean
        sampled_lat = selected_mu_lat
        sampled_lon = selected_mu_lon
    else:
        # Stochastic: sample from the selected Gaussian
        # Using reparameterisation: sample = mu + sigma * epsilon
        epsilon_lat = torch.randn_like(selected_mu_lat)
        epsilon_lon = torch.randn_like(selected_mu_lon)
        sampled_lat = selected_mu_lat + selected_sigma_lat * epsilon_lat
        sampled_lon = selected_mu_lon + selected_sigma_lon * epsilon_lon

    if return_sigma:
        return sampled_lat, sampled_lon, selected_sigma_lat, selected_sigma_lon
    return sampled_lat, sampled_lon
