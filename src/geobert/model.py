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
        bert_config = AutoConfig.from_pretrained(config.bert_model_name)
        self.bert = AutoModel.from_pretrained(config.bert_model_name)

        # Regression head: Linear(128->256) -> ReLU -> Linear(256->2)
        self.regression_head = nn.Sequential(
            nn.Linear(bert_config.hidden_size, config.hidden_dim),
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
