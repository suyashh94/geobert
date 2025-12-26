"""Training orchestration for GeoBERT model."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler

from geobert.config import ExperimentConfig
from geobert.device import get_device
from geobert.metrics import GeoMetrics, compute_metrics
from geobert.model import GeoBERTModel
from geobert.normalization import NormalizationStats

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from geobert.dataset import GeoBERTBatch


class Trainer:
    """Training orchestrator for GeoBERT model.

    Handles training loop, validation, checkpointing, and MLflow logging.
    Supports DistributedDataParallel (DDP) for multi-GPU training.

    :param model: GeoBERT model instance.
    :param train_loader: Training DataLoader.
    :param val_loader: Validation DataLoader.
    :param config: Experiment configuration.
    :param norm_stats: Normalization statistics.
    :param train_sampler: Optional DistributedSampler for DDP training.
    """

    def __init__(
        self,
        model: GeoBERTModel,
        train_loader: DataLoader[GeoBERTBatch],
        val_loader: DataLoader[GeoBERTBatch],
        config: ExperimentConfig,
        norm_stats: NormalizationStats,
        train_sampler: DistributedSampler[GeoBERTBatch] | None = None,
    ) -> None:
        self.config = config
        self.norm_stats = norm_stats
        self.train_sampler = train_sampler

        # Distributed training setup
        self.distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.is_main_process = self.rank == 0

        # Device setup for DDP (each process uses its local GPU)
        if self.distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = get_device()

        # Move model to device
        self.model: nn.Module = model.to(self.device)

        # Multi-GPU support with DistributedDataParallel
        if self.distributed:
            if self.is_main_process:
                print(f"Using {self.world_size} GPUs with DistributedDataParallel")
            # find_unused_parameters=True because BERT has pooler layers we don't use
            self.model = DDP(
                self.model,
                device_ids=[self.device],
                find_unused_parameters=True,
            )
        elif config.training.use_multi_gpu and torch.cuda.device_count() > 1:
            # Fallback to DataParallel for non-distributed multi-GPU (not recommended)
            if self.is_main_process:
                print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * config.training.num_epochs
        warmup_steps = int(total_steps * config.training.warmup_ratio)
        self.scheduler = self._create_scheduler(warmup_steps, total_steps)

        # Loss function
        self.criterion = nn.MSELoss()

        # State tracking
        self.global_step = 0
        self.best_val_loss = float("inf")

    def _create_scheduler(
        self,
        warmup_steps: int,
        total_steps: int,
    ) -> LambdaLR:
        """Create linear warmup + linear decay scheduler."""

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

        return LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch.

        :param epoch: Current epoch number (0-indexed).
        :return: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Set epoch for distributed sampler (ensures proper shuffling each epoch)
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(input_ids, attention_mask)
            loss = self.criterion(predictions, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Logging (only on main process)
            if self.global_step % self.config.training.log_interval == 0 and self.is_main_process:
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch + 1} | Step {self.global_step} | "
                    f"Loss: {loss.item():.6f} | LR: {lr:.2e}"
                )
                mlflow.log_metrics(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                    },
                    step=self.global_step,
                )

            # Intermediate validation (only on main process)
            if self.global_step % self.config.training.eval_interval == 0 and self.is_main_process:
                val_metrics = self.validate()
                self._log_validation_metrics(val_metrics)

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> GeoMetrics:
        """Run validation and compute metrics.

        :return: GeoMetrics for the validation set.
        """
        # Use unwrapped model for validation to avoid DDP synchronization issues
        # when only rank 0 runs validation
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.eval()
        all_predictions = []
        all_labels = []

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]

            predictions = model(input_ids, attention_mask)

            all_predictions.append(predictions.cpu())
            all_labels.append(labels)

        all_predictions_tensor = torch.cat(all_predictions, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)

        metrics = compute_metrics(all_predictions_tensor, all_labels_tensor, self.norm_stats)

        # Set back to train mode (use self.model so DDP wrapper is also in train mode)
        self.model.train()
        return metrics

    def _log_validation_metrics(self, metrics: GeoMetrics) -> None:
        """Log validation metrics to console and MLflow."""
        print(
            f"Validation | MSE: {metrics.mse:.6f} | "
            f"Mean Distance: {metrics.mean_distance_m:.1f}m | "
            f"Median Distance: {metrics.median_distance_m:.1f}m"
        )
        mlflow.log_metrics(
            {
                "val/mse": metrics.mse,
                "val/mae_lat": metrics.mae_lat,
                "val/mae_lon": metrics.mae_lon,
                "val/mean_distance_m": metrics.mean_distance_m,
                "val/median_distance_m": metrics.median_distance_m,
            },
            step=self.global_step,
        )

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> Path:
        """Save model checkpoint.

        :param epoch: Current epoch number.
        :param is_best: Whether this is the best model so far.
        :return: Path to saved checkpoint.
        """
        checkpoint_dir = self.config.training.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get the underlying model if using DataParallel
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }

        path = checkpoint_dir / "checkpoint_epoch.pt"
        torch.save(checkpoint, path)

        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            mlflow.log_artifact(str(best_path))

        return path

    def train(self) -> None:
        """Run full training loop."""
        if self.is_main_process:
            print(f"Training on device: {self.device}")
            print(f"Total training samples: {len(self.train_loader.dataset)}")  # type: ignore[arg-type]
            print(f"Total validation samples: {len(self.val_loader.dataset)}")  # type: ignore[arg-type]
            print(f"Batch size: {self.config.training.batch_size}")
            print(f"Steps per epoch: {len(self.train_loader)}")
            if self.distributed:
                print(f"World size: {self.world_size}")

        # Setup MLflow (only on main process)
        if self.is_main_process:
            mlflow.set_experiment(self.config.experiment_name)
            mlflow_run = mlflow.start_run(run_name=self.config.run_name)
            mlflow_run.__enter__()

            # Log configuration
            mlflow.log_params(
                {
                    "model_name": self.config.model.bert_model_name,
                    "hidden_dim": self.config.model.hidden_dim,
                    "batch_size": self.config.training.batch_size,
                    "learning_rate": self.config.training.learning_rate,
                    "num_epochs": self.config.training.num_epochs,
                    "max_seq_length": self.config.data.max_seq_length,
                    "world_size": self.world_size,
                }
            )

            # Save normalization stats
            norm_path = self.config.training.checkpoint_dir / "norm_stats.json"
            self.norm_stats.save(norm_path)
            mlflow.log_artifact(str(norm_path))

        try:
            for epoch in range(self.config.training.num_epochs):
                start_time = time.time()

                train_loss = self.train_epoch(epoch)

                # Validation and logging only on main process
                if self.is_main_process:
                    val_metrics = self.validate()
                    epoch_time = time.time() - start_time

                    print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
                    print(f"Train Loss: {train_loss:.6f}")
                    self._log_validation_metrics(val_metrics)
                    print(f"Epoch Time: {epoch_time:.1f}s\n")

                    # Check for best model
                    is_best = val_metrics.mse < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics.mse

                    # Save checkpoint
                    if (epoch + 1) % self.config.training.save_interval == 0:
                        self.save_checkpoint(epoch, is_best)

                # Synchronize all processes at end of epoch
                if self.distributed:
                    dist.barrier()

            if self.is_main_process:
                print("Training complete!")
                print(f"Best validation MSE: {self.best_val_loss:.6f}")
        finally:
            if self.is_main_process:
                mlflow_run.__exit__(None, None, None)
