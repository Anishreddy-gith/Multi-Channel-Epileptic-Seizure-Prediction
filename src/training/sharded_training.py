import os

import torch
from torch.utils.data import DataLoader

from src.training.sharded_dataset import (
    ShardBatchSampler,
    ShardedEEGDataset,
    discover_shards,
    load_split_metadata,
)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loader_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    return max(0, min(4, cpu_count - 1))


def create_sharded_loaders(
    batch_size: int,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, ShardedEEGDataset]]:
    shards = discover_shards()
    train_meta = load_split_metadata("train")
    val_meta = load_split_metadata("validation")
    test_meta = load_split_metadata("test")

    datasets = {
        "train": ShardedEEGDataset(train_meta, shards=shards),
        "validation": ShardedEEGDataset(val_meta, shards=shards),
        "test": ShardedEEGDataset(test_meta, shards=shards),
    }

    num_workers = loader_worker_count()
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_sampler = ShardBatchSampler(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_sampler = ShardBatchSampler(
        datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    test_sampler = ShardBatchSampler(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        common_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        datasets["train"],
        batch_sampler=train_sampler,
        **common_kwargs,
    )
    val_loader = DataLoader(
        datasets["validation"],
        batch_sampler=val_sampler,
        **common_kwargs,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_sampler=test_sampler,
        **common_kwargs,
    )

    print("\nSharded Dataset Sizes")
    print("Train:", len(datasets["train"]))
    print("Val  :", len(datasets["validation"]))
    print("Test :", len(datasets["test"]))
    print("Shards:", len(shards))
    print("DataLoader workers:", num_workers)
    print("Pinned memory:", pin_memory)

    return train_loader, val_loader, test_loader, datasets


def move_batch_to_device(xb, yb, device):
    non_blocking = device.type == "cuda"
    xb = xb.to(device, non_blocking=non_blocking)
    yb = yb.to(device, non_blocking=non_blocking)
    return xb, yb
