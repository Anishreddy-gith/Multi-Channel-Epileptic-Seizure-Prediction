from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, Dataset


INTERIM_DIR = Path("data/interim/windows")
SPLIT_DIR = Path("data/processed/splits")


@dataclass(frozen=True)
class ShardInfo:
    shard_id: int
    edf_name: str
    x_path: Path
    y_path: Path
    meta_path: Path
    n_rows: int
    shape: tuple[int, ...]
    dtype: np.dtype


def _metadata_row_count(meta_path: Path) -> int:
    total = 0
    for chunk in pd.read_csv(meta_path, chunksize=100_000):
        total += len(chunk)
    return total


def discover_shards(interim_dir: Path = INTERIM_DIR) -> list[ShardInfo]:
    if not interim_dir.exists():
        raise FileNotFoundError(f"Shard directory not found: {interim_dir}")

    x_files = sorted(interim_dir.glob("*_X.npy"))
    if not x_files:
        raise FileNotFoundError(f"No *_X.npy shards found in {interim_dir}")

    shards: list[ShardInfo] = []
    for shard_id, x_path in enumerate(x_files):
        edf_name = x_path.name[: -len("_X.npy")]
        y_path = interim_dir / f"{edf_name}_y.npy"
        meta_path = interim_dir / f"{edf_name}_meta.csv"

        missing = [path for path in (y_path, meta_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing shard companion(s) for {x_path.name}: "
                f"{[str(path) for path in missing]}"
            )

        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        if X.ndim != 3:
            raise ValueError(f"{x_path} must be 3D, got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"{y_path} must be 1D, got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Shard row mismatch for {edf_name}: "
                f"X={X.shape[0]}, y={y.shape[0]}"
            )

        shards.append(
            ShardInfo(
                shard_id=shard_id,
                edf_name=edf_name,
                x_path=x_path,
                y_path=y_path,
                meta_path=meta_path,
                n_rows=int(X.shape[0]),
                shape=tuple(X.shape[1:]),
                dtype=np.dtype(X.dtype),
            )
        )

        del X
        del y

    feature_shapes = {shard.shape for shard in shards}
    if len(feature_shapes) != 1:
        raise ValueError(f"Inconsistent shard feature shapes: {feature_shapes}")

    return shards


def load_shard_metadata(shards: list[ShardInfo]) -> pd.DataFrame:
    parts = []
    for shard in shards:
        meta = pd.read_csv(shard.meta_path)
        required = {"patient_id", "file_name", "label"}
        missing = required.difference(meta.columns)
        if missing:
            raise ValueError(f"{shard.meta_path} is missing columns: {sorted(missing)}")

        if len(meta) != shard.n_rows:
            raise ValueError(
                f"Metadata length mismatch for {shard.edf_name}: "
                f"{len(meta)} != {shard.n_rows}"
            )

        meta = meta.copy()
        meta["shard_id"] = shard.shard_id
        meta["shard_row"] = np.arange(len(meta), dtype=np.int64)
        parts.append(meta)

    if not parts:
        raise ValueError("No shard metadata loaded")

    metadata = pd.concat(parts, ignore_index=True)
    metadata["dataset_index"] = np.arange(len(metadata), dtype=np.int64)
    metadata["patient_id"] = metadata["patient_id"].astype(str)
    metadata["file_name"] = metadata["file_name"].astype(str)
    metadata["label"] = metadata["label"].astype(int)
    return metadata


def load_split_metadata(split_name: str, split_dir: Path = SPLIT_DIR) -> pd.DataFrame:
    candidates = {
        "train": [split_dir / "train_metadata.csv"],
        "validation": [
            split_dir / "validation_metadata.csv",
            split_dir / "val_metadata.csv",
        ],
        "val": [
            split_dir / "validation_metadata.csv",
            split_dir / "val_metadata.csv",
        ],
        "test": [split_dir / "test_metadata.csv"],
    }
    if split_name not in candidates:
        raise ValueError(f"Unknown split name: {split_name}")

    for path in candidates[split_name]:
        if path.exists():
            return pd.read_csv(path)

    raise FileNotFoundError(
        f"No split CSV found for {split_name}: {[str(p) for p in candidates[split_name]]}"
    )


class ShardedEEGDataset(Dataset):
    def __init__(
        self,
        split_metadata: pd.DataFrame,
        shards: list[ShardInfo] | None = None,
        add_channel_dim: bool = True,
    ) -> None:
        self.shards = shards if shards is not None else discover_shards()
        self.shard_by_id = {shard.shard_id: shard for shard in self.shards}
        self.add_channel_dim = add_channel_dim

        required = {"shard_id", "shard_row", "label"}
        missing = required.difference(split_metadata.columns)
        if missing:
            raise ValueError(f"Split metadata is missing columns: {sorted(missing)}")

        self.index = split_metadata[["shard_id", "shard_row", "label"]].copy()
        self.index["shard_id"] = self.index["shard_id"].astype(np.int64)
        self.index["shard_row"] = self.index["shard_row"].astype(np.int64)
        self.index["label"] = self.index["label"].astype(np.int64)
        self.index.reset_index(drop=True, inplace=True)
        self._shard_ids = self.index["shard_id"].to_numpy(dtype=np.int64)
        self._shard_rows = self.index["shard_row"].to_numpy(dtype=np.int64)
        self._labels = self.index["label"].to_numpy(dtype=np.int64)

        self._open_shard_id: int | None = None
        self._open_X = None
        self._open_y = None

    def __len__(self) -> int:
        return len(self.index)

    def _open_shard(self, shard_id: int) -> None:
        if self._open_shard_id == shard_id:
            return

        self._open_X = None
        self._open_y = None
        shard = self.shard_by_id[int(shard_id)]
        self._open_X = np.load(shard.x_path, mmap_mode="r")
        self._open_y = np.load(shard.y_path, mmap_mode="r")
        self._open_shard_id = int(shard_id)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_id = int(self._shard_ids[index])
        shard_row = int(self._shard_rows[index])

        self._open_shard(shard_id)
        x = np.asarray(self._open_X[shard_row], dtype=np.float32)
        y = int(self._open_y[shard_row])
        expected_y = int(self._labels[index])
        if y != expected_y:
            raise RuntimeError(
                f"Label mismatch at dataset index {index}: "
                f"split={expected_y}, shard={y}"
            )

        if self.add_channel_dim:
            x = x[None, :, :]

        return torch.from_numpy(x.copy()), torch.tensor(y, dtype=torch.long)

    def close(self) -> None:
        self._open_X = None
        self._open_y = None
        self._open_shard_id = None


class ShardBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: ShardedEEGDataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        grouped: dict[int, list[int]] = {}
        for global_index, shard_id in enumerate(dataset.index["shard_id"].to_numpy()):
            grouped.setdefault(int(shard_id), []).append(global_index)
        self.grouped_indices = grouped

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        shard_ids = list(self.grouped_indices)
        if self.shuffle:
            rng.shuffle(shard_ids)

        for shard_id in shard_ids:
            indices = np.asarray(self.grouped_indices[shard_id], dtype=np.int64)
            if self.shuffle:
                rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size].tolist()
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

        self.epoch += 1

    def __len__(self) -> int:
        total = 0
        for indices in self.grouped_indices.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total
