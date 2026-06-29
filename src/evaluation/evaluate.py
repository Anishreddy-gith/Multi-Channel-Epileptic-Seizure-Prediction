from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.training.losses_metrics import binary_classification_metrics, optimize_threshold
from src.training.sharded_dataset import ShardBatchSampler, ShardedEEGDataset, discover_shards
from src.training.sharded_training import get_device
from src.training.train import build_model, collect_probabilities, load_checkpoint
from src.utils.project_paths import artifact_path, ensure_project_storage, print_storage_summary, project_path


def load_split(path: str | Path) -> pd.DataFrame:
    split = pd.read_csv(path)
    required = {"shard_id", "shard_row", "label"}
    missing = required.difference(split.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return split


def make_loader(split: pd.DataFrame, batch_size: int, num_workers: int) -> DataLoader:
    shards = discover_shards()
    dataset = ShardedEEGDataset(split, shards=shards)
    sampler = ShardBatchSampler(dataset, batch_size=batch_size, shuffle=False)
    kwargs = {
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def evaluate_checkpoint(
    model_name: str,
    checkpoint_path: str | Path,
    split_path: str | Path,
    threshold: float | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict:
    device = get_device()
    model = build_model(model_name).to(device)
    checkpoint_path = project_path(checkpoint_path)
    split_path = project_path(split_path)
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)

    if threshold is None:
        threshold = checkpoint.get("best_threshold", None)

    loader = make_loader(load_split(split_path), batch_size, num_workers)
    y_true, y_prob = collect_probabilities(
        model,
        loader,
        device,
        use_amp=device.type == "cuda",
    )

    if threshold is None:
        threshold, _ = optimize_threshold(y_true, y_prob, metric="score")

    metrics = binary_classification_metrics(y_true, y_prob, threshold)
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["split"] = str(split_path)
    metrics["model"] = model_name
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["eegnet", "attention", "hybrid"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="data/processed/splits/test_metadata.csv")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main():
    ensure_project_storage()
    args = parse_args()
    storage = {
        "checkpoint": project_path(args.checkpoint),
        "split": project_path(args.split),
    }
    if args.output:
        storage["metrics_output"] = artifact_path(args.output)
    print_storage_summary(storage)
    metrics = evaluate_checkpoint(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        split_path=args.split,
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(metrics, indent=2))

    if args.output:
        output_path = artifact_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
