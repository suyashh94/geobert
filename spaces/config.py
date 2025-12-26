"""Configuration dataclasses for GeoBERT inference."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the GeoBERT model architecture.

    :param bert_model_name: HuggingFace model identifier for BERT.
    :param hidden_dim: Hidden dimension of the regression head.
    :param output_dim: Output dimension (2 for lat/long).
    """

    bert_model_name: str = "google/bert_uncased_L-2_H-128_A-2"
    hidden_dim: int = 256
    output_dim: int = 2
