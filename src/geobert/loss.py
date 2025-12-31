import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNLoss(nn.Module):
    def __init__(self, sigma_min: float = 1e-6):
        super().__init__()
        self.sigma_min = sigma_min

    def forward(
        self,
        pi_logits: torch.Tensor,
        mu_lat: torch.Tensor,
        mu_lon: torch.Tensor,
        sigma_lat: torch.Tensor,
        sigma_lon: torch.Tensor,
        target_lat: torch.Tensor,
        target_lon: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Mixture Density Network loss.

        :param pi_logits: Mixture coefficients of shape (batch, num_mixtures).
        :param mu_lat: Means for latitude of shape (batch, num_mixtures).
        :param mu_lon: Means for longitude of shape (batch, num_mixtures).
        :param sigma_lat: Standard deviations for latitude of shape (batch, num_mixtures).
        :param sigma_lon: Standard deviations for longitude of shape (batch, num_mixtures).
        :param target_lat: True latitude values of shape (batch,).
        :param target_lon: True longitude values of shape (batch,).
        :return: Computed MDN loss as a scalar tensor.
        """
        batch_size, num_mixtures = pi_logits.size()

        sigma_lat = torch.clamp(sigma_lat, min=self.sigma_min)
        sigma_lon = torch.clamp(sigma_lon, min=self.sigma_min)

        # Expand target coordinates to match mixture components
        target_lat = target_lat.unsqueeze(1).expand(-1, num_mixtures)  # (batch, K)
        target_lon = target_lon.unsqueeze(1).expand(-1, num_mixtures)  # (batch, K)

        log_pi = F.log_softmax(pi_logits, dim=1)

        log_prob_lat = (
            -0.5 * torch.log(torch.tensor(2 * torch.pi, device=mu_lat.device))
            - torch.log(sigma_lat)
            - 0.5 * ((target_lat - mu_lat) / sigma_lat) ** 2
        )  # (batch, K)

        log_prob_lon = (
            -0.5 * torch.log(torch.tensor(2 * torch.pi, device=mu_lon.device))
            - torch.log(sigma_lon)
            - 0.5 * ((target_lon - mu_lon) / sigma_lon) ** 2
        )  # (batch, K)

        # Joint log probability
        log_prob_joint = log_prob_lat + log_prob_lon  # (batch, K)
        # Weighted by mixture coefficients
        log_weighted = log_pi + log_prob_joint  # (batch, K)
        # log likelihood
        log_likelihood = torch.logsumexp(log_weighted, dim=1)  # (batch,)
        loss = -torch.mean(log_likelihood)
        return loss
