"""Configuration dataclasses for GeoBERT training."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data loading and preprocessing.

    :param data_path: Path to the parquet file with address data.
    :param train_ratio: Fraction of data for training.
    :param val_ratio: Fraction of data for validation.
    :param test_ratio: Fraction of data for testing.
    :param random_seed: Seed for reproducible data splits.
    :param max_seq_length: Maximum token sequence length for BERT.
    :param num_samples: Limit total samples (None for all data).
    """

    data_path: Path = Path("data/nyc_geocoding_processed.parquet")
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    max_seq_length: int = 32
    num_samples: int | None = None


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


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training hyperparameters.

    :param batch_size: Training batch size per GPU.
    :param num_epochs: Number of training epochs.
    :param learning_rate: AdamW learning rate.
    :param weight_decay: AdamW weight decay.
    :param warmup_ratio: Fraction of steps for learning rate warmup.
    :param num_workers: Number of DataLoader workers.
    :param checkpoint_dir: Directory for saving model checkpoints.
    :param log_interval: Steps between logging.
    :param eval_interval: Steps between validation runs.
    :param save_interval: Epochs between checkpoint saves.
    :param use_multi_gpu: Enable DataParallel for multi-GPU training.
    :param max_grad_norm: Maximum gradient norm for clipping.
    """

    batch_size: int = 256
    num_epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_workers: int = 4
    checkpoint_dir: Path = Path("outputs/checkpoints")
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 1
    use_multi_gpu: bool = True
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration combining all sub-configs.

    :param data: Data loading configuration.
    :param model: Model architecture configuration.
    :param training: Training hyperparameters.
    :param experiment_name: Name for MLflow experiment.
    :param run_name: Name for this specific run.
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: str = "geobert-training"
    run_name: str | None = None
