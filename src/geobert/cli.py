"""CLI entry points for GeoBERT training and evaluation."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from geobert.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)
from geobert.dataset import create_data_splits, create_dataloaders
from geobert.device import print_device_info
from geobert.model import GeoBERTModel
from geobert.trainer import Trainer


def train() -> None:
    """CLI entry point for geobert-train command."""
    parser = argparse.ArgumentParser(description="Train GeoBERT geocoding model")

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/nyc_geocoding_processed.parquet"),
        help="Path to processed parquet data file",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=32,
        help="Maximum token sequence length",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=None,
        help="Limit total samples (default: use all data)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: use 1000 samples, 1 epoch, batch_size=32",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )

    # Output arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/checkpoints"),
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="geobert-training",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name",
    )

    # Misc
    parser.add_argument(
        "--no-multi-gpu",
        action="store_true",
        help="Disable multi-GPU training",
    )

    args = parser.parse_args()

    # Apply debug mode overrides
    if args.debug:
        print("DEBUG MODE: Using 1000 samples, 1 epoch, batch_size=32")
        args.num_samples = 1000
        args.epochs = 2
        args.batch_size = 32

    # Print device info
    print_device_info()

    # Build configuration
    config = ExperimentConfig(
        data=DataConfig(
            data_path=args.data_path,
            max_seq_length=args.max_seq_length,
            num_samples=args.num_samples,
        ),
        model=ModelConfig(),
        training=TrainingConfig(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
            checkpoint_dir=args.checkpoint_dir,
            use_multi_gpu=not args.no_multi_gpu,
        ),
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )

    # Load tokenizer
    print(f"Loading tokenizer: {config.model.bert_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.bert_model_name)

    # Create datasets
    print(f"Loading data from: {config.data.data_path}")
    train_dataset, val_dataset, test_dataset, norm_stats = create_data_splits(
        config.data, tokenizer
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        config.training.batch_size,
        config.training.num_workers,
    )

    # Create model
    print("Creating model...")
    model = GeoBERTModel(config.model)
    param_counts = model.get_num_parameters()
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")

    # Create trainer and train
    trainer = Trainer(model, train_loader, val_loader, config, norm_stats)
    trainer.train()


def evaluate() -> None:
    """CLI entry point for geobert-eval command."""
    raise NotImplementedError("Evaluation not yet implemented")


if __name__ == "__main__":
    train()
