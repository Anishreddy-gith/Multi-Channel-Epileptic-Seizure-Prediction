from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.project_paths import (  # noqa: E402
    ensure_project_storage,
    experiment_path,
    output_path,
    print_storage_summary,
)

ensure_project_storage()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.training.losses_metrics import (  # noqa: E402
    binary_classification_metrics,
    build_loss,
    compute_class_weights_from_labels,
    optimize_threshold,
)
from src.training.sharded_dataset import (  # noqa: E402
    ShardBatchSampler,
    ShardedEEGDataset,
    discover_shards,
    load_split_metadata,
)
from src.training.sharded_training import get_device, move_batch_to_device  # noqa: E402
from src.training.train import build_model, checkpoint_state, load_checkpoint, set_deterministic_seed  # noqa: E402


OUTPUT_DIR = output_path("paper/class_imbalance_study")
CHECKPOINT_DIR = experiment_path("checkpoints/class_imbalance_study")
DATA_AUDIT_REPORT = output_path("paper/data_audit_report.json")

METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "mcc",
    "cohen_kappa",
    "sensitivity",
    "specificity",
]


@dataclass(frozen=True)
class Strategy:
    strategy_id: str
    display_name: str
    base_id: str
    loss_name: str
    class_weights: str | None
    sampler: str
    threshold_mode: str


EEGNET_STRATEGIES = [
    Strategy(
        "weighted_ce",
        "Weighted CrossEntropy",
        "weighted_ce",
        "weighted_ce",
        "auto",
        "shard",
        "fixed",
    ),
    Strategy(
        "focal_loss",
        "Focal Loss",
        "focal_loss",
        "focal",
        "auto",
        "shard",
        "fixed",
    ),
    Strategy(
        "weighted_random_sampler",
        "WeightedRandomSampler",
        "weighted_random_sampler",
        "cross_entropy",
        None,
        "weighted_random",
        "fixed",
    ),
    Strategy(
        "weighted_ce_threshold",
        "Weighted CrossEntropy + Threshold Optimization",
        "weighted_ce",
        "weighted_ce",
        "auto",
        "shard",
        "optimized",
    ),
    Strategy(
        "focal_loss_threshold",
        "Focal Loss + Threshold Optimization",
        "focal_loss",
        "focal",
        "auto",
        "shard",
        "optimized",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--followup-models", nargs="*", default=["attention", "hybrid"])
    return parser.parse_args()


def as_jsonable(value):
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [as_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return None if not math.isfinite(out) else out
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def verify_data_audit_passed() -> None:
    if not DATA_AUDIT_REPORT.exists():
        raise FileNotFoundError(f"Missing data audit report: {DATA_AUDIT_REPORT}")
    with open(DATA_AUDIT_REPORT, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "PASS" or report.get("issues"):
        raise RuntimeError("Data audit did not pass; refusing to start training.")


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def loader_kwargs(num_workers: int, prefetch_factor: int) -> dict:
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def create_loaders(strategy: Strategy, args: argparse.Namespace):
    shards = discover_shards()
    train_dataset = ShardedEEGDataset(load_split_metadata("train"), shards=shards)
    val_dataset = ShardedEEGDataset(load_split_metadata("validation"), shards=shards)
    test_dataset = ShardedEEGDataset(load_split_metadata("test"), shards=shards)
    common = loader_kwargs(args.num_workers, args.prefetch_factor)

    if strategy.sampler == "weighted_random":
        labels = train_dataset._labels
        class_counts = np.bincount(labels, minlength=2).astype(np.float64)
        sample_weights = 1.0 / class_counts[labels]
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            drop_last=False,
            **common,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=ShardBatchSampler(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                seed=args.seed,
            ),
            **common,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=ShardBatchSampler(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
        ),
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=ShardBatchSampler(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
        ),
        **common,
    )
    return train_loader, val_loader, test_loader


def collect_probabilities(model, loader, device, use_amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels = []
    probs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = move_batch_to_device(xb, yb, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(xb)
                batch_probs = torch.softmax(logits, dim=1)[:, 1]
            labels.append(yb.detach().cpu().numpy())
            probs.append(batch_probs.detach().cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs)


def metric_score(metrics: dict) -> float:
    return float(metrics["pr_auc"]) + float(metrics["f1"])


def evaluate_strategy(strategy: Strategy, y_true, y_prob) -> tuple[float, dict]:
    if strategy.threshold_mode == "optimized":
        threshold, metrics = optimize_threshold(y_true, y_prob, metric="f1")
    else:
        threshold = 0.5
        metrics = binary_classification_metrics(y_true, y_prob, threshold)
    return float(threshold), metrics


def state_for_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_score,
    threshold,
    config,
) -> dict:
    return checkpoint_state(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        best_score,
        threshold,
        config,
    )


def maybe_load_last(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    args: argparse.Namespace,
):
    if not args.resume or not path.exists():
        return 0, {}, [], 0
    checkpoint = load_checkpoint(
        path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
    )
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_scores = {k: float(v) for k, v in checkpoint.get("best_scores", {}).items()}
    history = checkpoint.get("history", [])
    bad_epochs = int(checkpoint.get("bad_epochs", 0))
    print(f"Resumed {path} at epoch {start_epoch}", flush=True)
    return start_epoch, best_scores, history, bad_epochs


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(as_jsonable(payload), handle, indent=2)


def plot_strategy_artifacts(strategy_dir: Path, strategy_name: str, y_true, y_prob, threshold, history_df):
    strategy_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["interictal", "preictal"]).plot(values_format="d")
    plt.title(strategy_name)
    plt.tight_layout()
    plt.savefig(strategy_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_score(y_true, y_prob)).plot()
    plt.title(strategy_name)
    plt.tight_layout()
    plt.savefig(strategy_dir / "roc_curve.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=pr_auc).plot()
    plt.title(strategy_name)
    plt.tight_layout()
    plt.savefig(strategy_dir / "precision_recall_curve.png", dpi=160)
    plt.close()

    if len(history_df):
        plt.figure(figsize=(9, 5))
        plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
        if "val_pr_auc" in history_df:
            plt.plot(history_df["epoch"], history_df["val_pr_auc"], label="val_pr_auc")
        if "val_f1" in history_df:
            plt.plot(history_df["epoch"], history_df["val_f1"], label="val_f1")
        if "val_balanced_accuracy" in history_df:
            plt.plot(history_df["epoch"], history_df["val_balanced_accuracy"], label="val_balanced_accuracy")
        plt.xlabel("epoch")
        plt.title(strategy_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(strategy_dir / "training_curves.png", dpi=160)
        plt.close()


def train_base_model(
    model_name: str,
    base_strategy: Strategy,
    variants: list[Strategy],
    args: argparse.Namespace,
) -> list[dict]:
    run_id = f"{model_name}_{base_strategy.base_id}"
    run_dir = OUTPUT_DIR / run_id
    ckpt_dir = CHECKPOINT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / "last.pth"

    if args.force:
        for path in [run_dir, ckpt_dir]:
            if path.exists():
                shutil.rmtree(path)
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": model_name,
        "base_strategy": base_strategy.base_id,
        "loss": base_strategy.loss_name,
        "sampler": base_strategy.sampler,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
        "early_stopping_patience": args.early_stopping_patience,
        "gradient_clip_norm": args.gradient_clip_norm,
        "focal_gamma": args.focal_gamma,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "amp": not args.no_amp,
        "class_weights": base_strategy.class_weights,
        "variants": [v.strategy_id for v in variants],
    }
    save_json(run_dir / "config.json", config)

    set_deterministic_seed(args.seed)
    device = get_device()
    use_amp = (not args.no_amp) and device.type == "cuda"
    train_loader, val_loader, test_loader = create_loaders(base_strategy, args)

    model = build_model(model_name).to(device)
    if base_strategy.class_weights == "auto":
        class_weights = compute_class_weights_from_labels(train_loader.dataset._labels)
    else:
        class_weights = None
    criterion = build_loss(
        base_strategy.loss_name,
        class_weights=class_weights,
        focal_gamma=args.focal_gamma,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_scores, history, bad_epochs = maybe_load_last(
        last_path,
        model,
        optimizer,
        scheduler,
        scaler,
        device,
        args,
    )
    for variant in variants:
        best_scores.setdefault(variant.strategy_id, -float("inf"))

    if start_epoch >= args.epochs:
        print(f"{run_id} already reached {args.epochs} epochs; evaluating checkpoints.", flush=True)
    else:
        print(
            f"Training {run_id} on {device} | amp={use_amp} | "
            f"batch_size={args.batch_size} | train_batches={len(train_loader)}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = move_batch_to_device(xb, yb, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            if args.gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        val_true, val_prob = collect_probabilities(model, val_loader, device, use_amp)

        epoch_rows = []
        improved = False
        best_epoch_score = -float("inf")
        for variant in variants:
            threshold, val_metrics = evaluate_strategy(variant, val_true, val_prob)
            score = metric_score(val_metrics)
            best_epoch_score = max(best_epoch_score, score)
            row = {
                "epoch": epoch + 1,
                "model": model_name,
                "base_strategy": base_strategy.base_id,
                "strategy_id": variant.strategy_id,
                "threshold": threshold,
                "train_loss": train_loss,
                **{f"val_{k}": val_metrics[k] for k in METRIC_COLUMNS},
                "selection_score": score,
            }
            epoch_rows.append(row)
            if score > best_scores[variant.strategy_id]:
                improved = True
                best_scores[variant.strategy_id] = score
                torch.save(
                    {
                        **state_for_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            epoch,
                            score,
                            threshold,
                            {**config, "strategy_id": variant.strategy_id},
                        ),
                        "best_scores": best_scores,
                        "history": history + epoch_rows,
                        "bad_epochs": 0,
                    },
                    ckpt_dir / f"{variant.strategy_id}_best.pth",
                )

        scheduler.step(best_epoch_score)
        bad_epochs = 0 if improved else bad_epochs + 1
        history.extend(epoch_rows)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        torch.save(
            {
                **state_for_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_epoch_score,
                    0.5,
                    config,
                ),
                "best_scores": best_scores,
                "history": history,
                "bad_epochs": bad_epochs,
            },
            last_path,
        )
        row_msg = " | ".join(
            f"{row['strategy_id']}: val_pr_auc={row['val_pr_auc']:.4f} val_f1={row['val_f1']:.4f} thr={row['threshold']:.2f}"
            for row in epoch_rows
        )
        print(
            f"{run_id} epoch {epoch + 1:03d}/{args.epochs} "
            f"loss={train_loss:.4f} {row_msg} "
            f"time={time.time() - epoch_start:.1f}s",
            flush=True,
        )
        if bad_epochs >= args.early_stopping_patience:
            print(f"Early stopping {run_id} at epoch {epoch + 1}", flush=True)
            break

    results = []
    history_df = pd.DataFrame(history)
    for variant in variants:
        ckpt_path = ckpt_dir / f"{variant.strategy_id}_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing best checkpoint for {variant.strategy_id}: {ckpt_path}")
        eval_model = build_model(model_name).to(device)
        checkpoint = load_checkpoint(ckpt_path, eval_model, device=device)
        threshold = float(checkpoint.get("best_threshold", 0.5))
        val_true, val_prob = collect_probabilities(eval_model, val_loader, device, use_amp)
        test_true, test_prob = collect_probabilities(eval_model, test_loader, device, use_amp)
        val_metrics = binary_classification_metrics(val_true, val_prob, threshold)
        test_metrics = binary_classification_metrics(test_true, test_prob, threshold)
        strategy_dir = OUTPUT_DIR / variant.strategy_id if model_name == "eegnet" else OUTPUT_DIR / f"{model_name}_{variant.strategy_id}"
        strategy_history = history_df[history_df["strategy_id"] == variant.strategy_id].copy()
        plot_strategy_artifacts(
            strategy_dir,
            f"{model_name}: {variant.display_name}",
            test_true,
            test_prob,
            threshold,
            strategy_history,
        )
        payload = {
            "model": model_name,
            "strategy_id": variant.strategy_id,
            "display_name": variant.display_name,
            "base_strategy": base_strategy.base_id,
            "checkpoint": str(ckpt_path),
            "threshold": threshold,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        save_json(strategy_dir / "metrics.json", payload)
        results.append(payload)
    return results


def comparison_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        row = {
            "model": result["model"],
            "strategy_id": result["strategy_id"],
            "strategy": result["display_name"],
            "base_strategy": result["base_strategy"],
            "threshold": result["threshold"],
            "checkpoint": result["checkpoint"],
        }
        for prefix, metrics in [
            ("validation", result["validation_metrics"]),
            ("test", result["test_metrics"]),
        ]:
            for metric in METRIC_COLUMNS:
                row[f"{prefix}_{metric}"] = metrics[metric]
        rows.append(row)
    return rows


def select_best_eegnet(results: list[dict]) -> dict:
    eegnet_results = [r for r in results if r["model"] == "eegnet"]
    if not eegnet_results:
        raise RuntimeError("No EEGNet results available for selection")
    return sorted(
        eegnet_results,
        key=lambda r: (
            float(r["validation_metrics"]["pr_auc"]),
            float(r["validation_metrics"]["f1"]),
            float(r["validation_metrics"]["balanced_accuracy"]),
        ),
        reverse=True,
    )[0]


def write_comparison(results: list[dict], best: dict | None = None) -> None:
    rows = comparison_rows(results)
    table = pd.DataFrame(rows)
    table.sort_values(
        ["model", "validation_pr_auc", "validation_f1"],
        ascending=[True, False, False],
        inplace=True,
    )
    table.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)
    table.to_markdown(OUTPUT_DIR / "comparison_table.md", index=False)
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selection_rule": "Highest validation PR-AUC, tie-broken by validation F1-score and balanced accuracy",
        "recommended_eegnet_strategy": best,
        "results": results,
    }
    save_json(OUTPUT_DIR / "summary.json", summary)


def main() -> None:
    args = parse_args()
    verify_data_audit_passed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print_storage_summary(
        {
            "study_outputs": OUTPUT_DIR,
            "study_checkpoints": CHECKPOINT_DIR,
            "data_audit_report": DATA_AUDIT_REPORT,
        }
    )
    if args.force and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "run_config.json", vars(args))

    base_to_variants: dict[str, list[Strategy]] = {}
    for strategy in EEGNET_STRATEGIES:
        base_to_variants.setdefault(strategy.base_id, []).append(strategy)

    all_results: list[dict] = []
    for base_id, variants in base_to_variants.items():
        all_results.extend(train_base_model("eegnet", variants[0], variants, args))
        best_so_far = select_best_eegnet(all_results)
        write_comparison(all_results, best_so_far)

    best = select_best_eegnet(all_results)
    write_comparison(all_results, best)
    print(
        "Recommended EEGNet imbalance strategy: "
        f"{best['display_name']} "
        f"(val_pr_auc={best['validation_metrics']['pr_auc']:.4f}, "
        f"val_f1={best['validation_metrics']['f1']:.4f})",
        flush=True,
    )

    best_strategy = next(s for s in EEGNET_STRATEGIES if s.strategy_id == best["strategy_id"])
    for model_name in args.followup_models:
        all_results.extend(train_base_model(model_name, best_strategy, [best_strategy], args))
        write_comparison(all_results, best)

    final_best = select_best_eegnet(all_results)
    write_comparison(all_results, final_best)
    print(f"Study complete. Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
