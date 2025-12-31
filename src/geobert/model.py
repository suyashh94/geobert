"""GeoBERT model for geocoding addresses to coordinates."""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from geobert.config import ModelConfig


class GeoBERTModel(nn.Module):
    """BERT-based geocoding model for predicting lat/long from addresses.

    Uses a pretrained tiny BERT model with a regression head.
    Architecture:
        BERT CLS embedding (128) -> Linear(256) -> ReLU -> Linear(2)

    :param config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Load pretrained BERT
        self.bert_config = AutoConfig.from_pretrained(config.bert_model_name)
        self.bert = AutoModel.from_pretrained(config.bert_model_name)

        # Regression head: Linear(128->256) -> ReLU -> Linear(256->2)
        self.regression_head = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through BERT and regression head.

        :param input_ids: Token IDs of shape (batch, seq_len).
        :param attention_mask: Attention mask of shape (batch, seq_len).
        :return: Predictions of shape (batch, 2) for [lat, lon].
        """
        # Get BERT outputs - attention mask ensures padding doesn't affect CLS
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Extract CLS token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, 128)

        # Regression head
        predictions = self.regression_head(cls_embedding)  # (batch, 2)

        return predictions

    def get_num_parameters(self) -> dict[str, int]:
        """Count trainable and total parameters.

        :return: Dictionary with parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# MDN implementation of GeoBertModel
class GeoBERTMDNModel(GeoBERTModel):
    """GeoBERT model with a Mixture Density Network (MDN) head for probabilistic geocoding.Parameters:
    config: Model configuration.
    num_mixtures: Number of Gaussian mixtures in the MDN head.
    """

    def __init__(self, config: ModelConfig, num_mixtures: int = 5) -> None:
        super().__init__(config)
        self.num_mixtures = num_mixtures

        self.pi_head = nn.Linear(self.bert_config.hidden_size, num_mixtures)
        self.mu_head = nn.Linear(self.bert_config.hidden_size, num_mixtures * config.output_dim)
        self.sigma_head = nn.Linear(self.bert_config.hidden_size, num_mixtures * config.output_dim)
        self.sigma_eps = 1e-6
        # Remove self.regresseion_head  property
        del self.regression_head

    def get_pi(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute mixture coefficients using softmax.

        :param logits: Raw logits of shape (batch, num_mixtures).
        :return: Mixture coefficients of shape (batch, num_mixtures).
        """
        pi = nn.Softmax(dim=-1)(logits)
        return pi

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through BERT and MDN head.

        :param input_ids: Token IDs of shape (batch, seq_len).
        :param attention_mask: Attention mask of shape (batch, seq_len).
        :return: Tuple of tensors for mixture coefficients (pi), means (mu_lat, mu_lon), and stddevs (sigma_lat, sigma_lon).
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Extract CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, 128)
        hidden = nn.ReLU()(cls_embedding)

        # Mixture coefficients
        pi_logits = self.pi_head(hidden)  # (batch, num_mixtures)

        # Means
        mu = self.mu_head(hidden)  # (batch, num_mixtures * 2)
        mu = mu.view(-1, self.num_mixtures, self.config.output_dim)  # (batch, num_mixtures, 2)
        mu_lat = mu[:, :, 0]  # (batch, num_mixtures)
        mu_lon = mu[:, :, 1]  # (batch, num_mixtures
        # Standard deviations
        sigma = self.sigma_head(hidden)  # (batch, num_mixtures * 2)
        sigma = sigma.view(
            -1, self.num_mixtures, self.config.output_dim
        )  # (batch, num_mixtures, 2)
        sigma_lat = nn.Softplus()(sigma[:, :, 0]) + self.sigma_eps  # (batch, num_mixtures)
        sigma_lon = nn.Softplus()(sigma[:, :, 1]) + self.sigma_eps  # (batch, num_mixtures)
        return pi_logits, mu_lat, mu_lon, sigma_lat, sigma_lon
