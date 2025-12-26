"""Dataset and DataLoader utilities for GeoBERT training."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from geobert.config import DataConfig
from geobert.normalization import NormalizationStats

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class GeoBERTBatch(TypedDict):
    """Type definition for a batch from GeoBERTDataset.

    :param input_ids: Token IDs tensor of shape (batch, seq_len).
    :param attention_mask: Attention mask tensor of shape (batch, seq_len).
    :param labels: Normalized [lat, lon] tensor of shape (batch, 2).
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class GeoBERTDataset(Dataset[GeoBERTBatch]):
    """PyTorch Dataset for GeoBERT geocoding model.

    Loads addresses and coordinates from parquet, tokenizes addresses
    with the BERT tokenizer, and normalizes coordinates.

    :param addresses: Series of address strings.
    :param latitudes: Series of latitude values.
    :param longitudes: Series of longitude values.
    :param tokenizer: HuggingFace tokenizer for BERT.
    :param max_seq_length: Maximum sequence length for tokenization.
    :param norm_stats: Normalization statistics for coordinates.
    """

    def __init__(
        self,
        addresses: pd.Series,  # type: ignore[type-arg]
        latitudes: pd.Series,  # type: ignore[type-arg]
        longitudes: pd.Series,  # type: ignore[type-arg]
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        norm_stats: NormalizationStats,
    ) -> None:
        self.addresses = addresses.reset_index(drop=True)
        self.latitudes = torch.tensor(latitudes.values, dtype=torch.float32)
        self.longitudes = torch.tensor(longitudes.values, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.norm_stats = norm_stats

        # Pre-compute normalized labels
        self.labels = norm_stats.normalize(self.latitudes, self.longitudes)

    def __len__(self) -> int:
        return len(self.addresses)

    def __getitem__(self, idx: int) -> GeoBERTBatch:
        address = self.addresses.iloc[idx]

        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            address,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": self.labels[idx],
        }


def create_data_splits(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[GeoBERTDataset, GeoBERTDataset, GeoBERTDataset, NormalizationStats]:
    """Create train/val/test datasets from parquet file.

    :param config: Data configuration.
    :param tokenizer: BERT tokenizer.
    :return: Tuple of (train_dataset, val_dataset, test_dataset, norm_stats).
    """
    df = pd.read_parquet(config.data_path)

    # Limit samples if specified
    if config.num_samples is not None:
        df = df.sample(n=min(config.num_samples, len(df)), random_state=config.random_seed)
        df = df.reset_index(drop=True)

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=config.train_ratio,
        random_state=config.random_seed,
    )

    # Second split: val vs test
    val_ratio_adjusted = config.val_ratio / (config.val_ratio + config.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_adjusted,
        random_state=config.random_seed,
    )

    # Compute normalization stats from TRAINING data only
    norm_stats = NormalizationStats.from_dataframe(train_df)

    # Create datasets
    train_dataset = GeoBERTDataset(
        train_df["address"],
        train_df["latitude"],
        train_df["longitude"],
        tokenizer,
        config.max_seq_length,
        norm_stats,
    )
    val_dataset = GeoBERTDataset(
        val_df["address"],
        val_df["latitude"],
        val_df["longitude"],
        tokenizer,
        config.max_seq_length,
        norm_stats,
    )
    test_dataset = GeoBERTDataset(
        test_df["address"],
        test_df["latitude"],
        test_df["longitude"],
        tokenizer,
        config.max_seq_length,
        norm_stats,
    )

    return train_dataset, val_dataset, test_dataset, norm_stats


def create_dataloaders(
    train_dataset: GeoBERTDataset,
    val_dataset: GeoBERTDataset,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader[GeoBERTBatch], DataLoader[GeoBERTBatch]]:
    """Create DataLoaders for training and validation.

    :param train_dataset: Training dataset.
    :param val_dataset: Validation dataset.
    :param batch_size: Batch size.
    :param num_workers: Number of worker processes.
    :return: Tuple of (train_loader, val_loader).
    """
    train_loader: DataLoader[GeoBERTBatch] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader: DataLoader[GeoBERTBatch] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
