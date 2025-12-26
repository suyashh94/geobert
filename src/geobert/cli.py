"""CLI entry points for GeoBERT training and evaluation."""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
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


def setup_distributed() -> tuple[bool, int]:
    """Initialize distributed training if launched with torchrun.

    :return: Tuple of (is_distributed, local_rank).
    """
    # Check if we're in a distributed environment (launched via torchrun)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

        # Set the device for this process
        torch.cuda.set_device(local_rank)

        return True, local_rank
    return False, 0


def cleanup_distributed() -> None:
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train() -> None:
    """CLI entry point for geobert-train command.

    Supports both single-GPU and multi-GPU training via torchrun:
        Single GPU: geobert-train [options]
        Multi-GPU:  torchrun --nproc_per_node=N -m geobert.cli [options]
    """
    # Initialize distributed training if applicable
    is_distributed, local_rank = setup_distributed()
    is_main_process = local_rank == 0 if is_distributed else True

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
        if is_main_process:
            if args.epochs is None:
                print("DEBUG MODE: Using 1000 samples, 2 epochs, batch_size=32")
            else:
                print(f"DEBUG MODE: Using 1000 samples, {args.epochs} epoch, batch_size=32")

        args.num_samples = 1000 if args.num_samples is None else args.num_samples
        args.epochs = 2 if args.epochs is None else args.epochs
        args.batch_size = 32

    # Print device info (only on main process)
    if is_main_process:
        print_device_info()
        if is_distributed:
            print(f"Distributed training enabled with {dist.get_world_size()} processes")

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
    if is_main_process:
        print(f"Loading tokenizer: {config.model.bert_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.bert_model_name)

    # Create datasets
    if is_main_process:
        print(f"Loading data from: {config.data.data_path}")
    train_dataset, val_dataset, test_dataset, norm_stats = create_data_splits(
        config.data, tokenizer
    )
    if is_main_process:
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        # Save test data for evaluation
        test_data_path = config.training.checkpoint_dir / "test_data.parquet"
        config.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        test_df = pd.DataFrame(
            {
                "address": test_dataset.addresses.tolist(),
                "latitude": test_dataset.latitudes.numpy(),
                "longitude": test_dataset.longitudes.numpy(),
            }
        )
        test_df.to_parquet(test_data_path)
        print(f"Saved test data to: {test_data_path}")

    # Create dataloaders (with DistributedSampler if distributed)
    train_loader, val_loader, train_sampler = create_dataloaders(
        train_dataset,
        val_dataset,
        config.training.batch_size,
        config.training.num_workers,
        distributed=is_distributed,
    )

    # Create model
    if is_main_process:
        print("Creating model...")
    model = GeoBERTModel(config.model)
    if is_main_process:
        param_counts = model.get_num_parameters()
        print(f"Total parameters: {param_counts['total']:,}")
        print(f"Trainable parameters: {param_counts['trainable']:,}")

    # Create trainer and train
    try:
        trainer = Trainer(model, train_loader, val_loader, config, norm_stats, train_sampler)
        trainer.train()
    finally:
        cleanup_distributed()


def evaluate() -> None:
    """CLI entry point for geobert-eval command."""
    raise NotImplementedError("Evaluation not yet implemented")


if __name__ == "__main__":
    train()
