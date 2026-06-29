from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.hybrid_attention import HybridEEGNetAttention
from src.models.hybrid_eegnet_transformer import HybridEEGNetTransformer
from src.training.losses_metrics import (
    binary_classification_metrics,
    build_loss,
    compute_class_weights_from_labels,
    optimize_threshold,
)
from src.training.sharded_dataset import (
    ShardBatchSampler,
    ShardedEEGDataset,
    discover_shards,
    load_split_metadata,
)
from src.training.sharded_training import get_device, move_batch_to_device
from src.utils.project_paths import ensure_project_storage, experiment_path, print_storage_summary, project_path


DEFAULT_CONFIG = {
    "model": "hybrid",
    "epochs": 25,
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "loss": "weighted_ce",
    "focal_gamma": 2.0,
    "class_weights": "auto",
    "scheduler": "plateau",
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,
    "early_stopping_patience": 5,
    "gradient_clip_norm": 1.0,
    "amp": True,
    "seed": 42,
    "num_workers": 4,
    "prefetch_factor": 2,
    "checkpoint_dir": str(experiment_path("checkpoints")),
    "run_dir": str(experiment_path("runs")),
    "resume": "",
    "threshold_metric": "score",
}


class EEGNetClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64), padding=(0, 32)),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=(18, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 1 * 256, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def load_config(path: str | Path | None, overrides: dict | None = None) -> dict:
    config = DEFAULT_CONFIG.copy()
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            user_config = json.load(handle)
        config.update(user_config)
    if overrides:
        config.update({k: v for k, v in overrides.items() if v is not None})
    return config


def build_model(model_name: str) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "eegnet":
        return EEGNetClassifier()
    if model_name in {"attention", "attn"}:
        return HybridEEGNetAttention()
    if model_name in {"hybrid", "transformer", "hybrid_transformer"}:
        return HybridEEGNetTransformer()
    raise ValueError(f"Unknown model: {model_name}")


def create_loaders_from_metadata(
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, ShardedEEGDataset]]:
    shards = discover_shards()
    datasets = {
        "train": ShardedEEGDataset(train_meta, shards=shards),
        "validation": ShardedEEGDataset(val_meta, shards=shards),
        "test": ShardedEEGDataset(test_meta, shards=shards),
    }

    num_workers = int(config["num_workers"])
    pin_memory = torch.cuda.is_available()
    common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        common_kwargs["prefetch_factor"] = int(config["prefetch_factor"])

    batch_size = int(config["batch_size"])
    seed = int(config["seed"])
    train_loader = DataLoader(
        datasets["train"],
        batch_sampler=ShardBatchSampler(datasets["train"], batch_size, True, seed=seed),
        **common_kwargs,
    )
    val_loader = DataLoader(
        datasets["validation"],
        batch_sampler=ShardBatchSampler(datasets["validation"], batch_size, False, seed=seed),
        **common_kwargs,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_sampler=ShardBatchSampler(datasets["test"], batch_size, False, seed=seed),
        **common_kwargs,
    )
    return train_loader, val_loader, test_loader, datasets


def create_default_loaders(config: dict):
    return create_loaders_from_metadata(
        load_split_metadata("train"),
        load_split_metadata("validation"),
        load_split_metadata("test"),
        config,
    )


def collect_probabilities(model, loader, device, use_amp: bool = False):
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = move_batch_to_device(xb, yb, device)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp and device.type == "cuda",
            ):
                logits = model(xb)
                batch_probs = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(batch_probs.detach().cpu().numpy())
            labels.extend(yb.detach().cpu().numpy())
    return np.asarray(labels), np.asarray(probs)


def checkpoint_state(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_score,
    best_threshold,
    config,
) -> dict:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_score": best_score,
        "best_threshold": best_threshold,
        "config": config,
    }


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint


def train_model(
    config: dict,
    train_meta: pd.DataFrame | None = None,
    val_meta: pd.DataFrame | None = None,
    test_meta: pd.DataFrame | None = None,
    run_name: str | None = None,
) -> dict:
    ensure_project_storage()
    set_deterministic_seed(int(config["seed"]))
    device = get_device()
    use_amp = bool(config["amp"]) and device.type == "cuda"

    if train_meta is None:
        train_loader, val_loader, test_loader, datasets = create_default_loaders(config)
    else:
        train_loader, val_loader, test_loader, datasets = create_loaders_from_metadata(
            train_meta,
            val_meta,
            test_meta,
            config,
        )

    model = build_model(config["model"]).to(device)

    if config["class_weights"] == "auto":
        class_weights = compute_class_weights_from_labels(train_loader.dataset._labels)
    elif config["class_weights"] in (None, "none", "None"):
        class_weights = None
    else:
        class_weights = np.asarray(config["class_weights"], dtype=np.float32)

    criterion = build_loss(
        config["loss"],
        class_weights=class_weights,
        focal_gamma=float(config["focal_gamma"]),
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"{timestamp}_{config['model']}"
    run_dir = project_path(config["run_dir"]) / run_name
    ckpt_dir = project_path(config["checkpoint_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print_storage_summary({"run_dir": run_dir, "checkpoint_dir": ckpt_dir})
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    history_rows = []
    start_epoch = 0
    best_score = -np.inf
    best_threshold = 0.5
    bad_epochs = 0

    if config.get("resume"):
        checkpoint = load_checkpoint(
            config["resume"],
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_score = float(checkpoint.get("best_score", best_score))
        best_threshold = float(checkpoint.get("best_threshold", best_threshold))

    best_path = ckpt_dir / f"{config['model']}_best.pth"
    last_path = ckpt_dir / f"{config['model']}_last.pth"

    for epoch in range(start_epoch, int(config["epochs"])):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb, yb = move_batch_to_device(xb, yb, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            if config["gradient_clip_norm"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config["gradient_clip_norm"]),
                )
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.detach().cpu())
            n_batches += 1

        train_loss /= max(n_batches, 1)
        val_true, val_prob = collect_probabilities(model, val_loader, device, use_amp)
        threshold, val_metrics = optimize_threshold(
            val_true,
            val_prob,
            metric=config["threshold_metric"],
        )
        score = 0.5 * val_metrics["f1"] + 0.5 * val_metrics["balanced_accuracy"]
        scheduler.step(score)

        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(run_dir / "metrics.csv", index=False)

        print(
            f"Epoch {epoch + 1:03d}/{config['epochs']} "
            f"loss={train_loss:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"thr={threshold:.2f} score={score:.4f}",
            flush=True,
        )

        state = checkpoint_state(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_score,
            best_threshold,
            config,
        )
        torch.save(state, last_path)

        if score > best_score:
            best_score = score
            best_threshold = threshold
            bad_epochs = 0
            state["best_score"] = best_score
            state["best_threshold"] = best_threshold
            torch.save(state, best_path)
        else:
            bad_epochs += 1

        if bad_epochs >= int(config["early_stopping_patience"]):
            print("Early stopping triggered.", flush=True)
            break

    load_checkpoint(best_path, model, device=device)
    test_true, test_prob = collect_probabilities(model, test_loader, device, use_amp)
    test_metrics = binary_classification_metrics(test_true, test_prob, best_threshold)
    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)

    print("Final test metrics")
    print(json.dumps(test_metrics, indent=2))
    return {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_score": best_score,
        "best_threshold": best_threshold,
        "test_metrics": test_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--model", choices=["eegnet", "attention", "hybrid"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss", choices=["weighted_ce", "cross_entropy", "focal"], default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "loss": args.loss,
        "resume": args.resume,
    }
    config = load_config(args.config or None, overrides)
    if args.no_amp:
        config["amp"] = False
    train_model(config)


if __name__ == "__main__":
    main()
