from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
LOGS_ROOT = PROJECT_ROOT / "logs"
CACHE_ROOT = LOGS_ROOT / "cache"
TMP_ROOT = LOGS_ROOT / "tmp"


def configure_project_environment() -> None:
    for path in [OUTPUTS_ROOT, EXPERIMENTS_ROOT, LOGS_ROOT, CACHE_ROOT, TMP_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
    env_paths = {
        "TMP": TMP_ROOT,
        "TEMP": TMP_ROOT,
        "TMPDIR": TMP_ROOT,
        "MPLCONFIGDIR": CACHE_ROOT / "matplotlib",
        "TORCH_HOME": CACHE_ROOT / "torch",
        "XDG_CACHE_HOME": CACHE_ROOT,
        "HF_HOME": CACHE_ROOT / "huggingface",
        "HF_DATASETS_CACHE": CACHE_ROOT / "huggingface" / "datasets",
        "TRANSFORMERS_CACHE": CACHE_ROOT / "huggingface" / "transformers",
        "WANDB_DIR": LOGS_ROOT / "wandb",
        "WANDB_CACHE_DIR": CACHE_ROOT / "wandb",
        "WANDB_CONFIG_DIR": CACHE_ROOT / "wandb_config",
        "MNE_DATA": CACHE_ROOT / "mne",
        "TORCHINDUCTOR_CACHE_DIR": CACHE_ROOT / "torchinductor",
        "CUDA_CACHE_PATH": CACHE_ROOT / "nvidia",
    }
    for key, path in env_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)


configure_project_environment()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

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
from src.training.sharded_training import move_batch_to_device  # noqa: E402
from src.training.train import build_model, checkpoint_state, load_checkpoint, set_deterministic_seed  # noqa: E402


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


WEIGHTED_CE = Strategy(
    "weighted_ce",
    "Weighted CrossEntropy",
    "weighted_ce",
    "weighted_ce",
    "auto",
    "shard",
    "fixed",
)
FOCAL = Strategy(
    "focal_loss",
    "Focal Loss",
    "focal_loss",
    "focal",
    "auto",
    "shard",
    "fixed",
)
WRS = Strategy(
    "weighted_random_sampler",
    "WeightedRandomSampler",
    "weighted_random_sampler",
    "cross_entropy",
    None,
    "weighted_random",
    "fixed",
)
WEIGHTED_CE_THRESHOLD = Strategy(
    "weighted_ce_threshold",
    "Weighted CrossEntropy + Threshold Optimization",
    "weighted_ce",
    "weighted_ce",
    "auto",
    "shard",
    "optimized",
)
FOCAL_THRESHOLD = Strategy(
    "focal_loss_threshold",
    "Focal Loss + Threshold Optimization",
    "focal_loss",
    "focal",
    "auto",
    "shard",
    "optimized",
)

STEP1_GROUPS = [
    (WEIGHTED_CE, [WEIGHTED_CE, WEIGHTED_CE_THRESHOLD]),
    (FOCAL, [FOCAL, FOCAL_THRESHOLD]),
    (WRS, [WRS]),
]
ALL_STEP1_STRATEGIES = [
    WEIGHTED_CE,
    FOCAL,
    WRS,
    WEIGHTED_CE_THRESHOLD,
    FOCAL_THRESHOLD,
]


class OvernightRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.started_at = time.time()
        self.run_id = args.run_id or time.strftime("overnight_%Y%m%d_%H%M%S")
        if args.smoke and not args.run_id:
            self.run_id = f"smoke_{time.strftime('%Y%m%d_%H%M%S')}"
        self.experiment_root = EXPERIMENTS_ROOT / self.run_id
        self.output_root = OUTPUTS_ROOT / "paper" / self.run_id
        self.config_root = self.experiment_root / "configs"
        self.checkpoint_root = self.experiment_root / "checkpoints"
        self.run_root = self.experiment_root / "runs"
        self.table_root = self.output_root / "tables"
        self.figure_root = self.output_root / "figures"
        self.metric_root = self.output_root / "metrics"
        self.prob_root = self.output_root / "probabilities"
        self.status_root = self.output_root / "status"
        self.progress_log = LOGS_ROOT / "progress.md"
        self.decision_log = LOGS_ROOT / "decision_log.md"
        self.stopped_log = LOGS_ROOT / "STOPPED.md"
        self.pid_file = LOGS_ROOT / "overnight_active.pid"
        self.manifest_path = self.experiment_root / "run_manifest.json"
        self.results: list[dict] = []
        self.training_records: list[dict] = []
        self.best_strategy: Strategy | None = None
        self.best_strategy_result: dict | None = None

    def now(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def append_log(self, path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line.rstrip() + "\n")

    def progress(self, step: str, event: str) -> None:
        self.append_log(self.progress_log, f"[{self.now()}] {step} — {event}")

    def decision(self, decision: str, reason: str) -> None:
        self.append_log(self.decision_log, f"[{self.now()}] DECISION — {decision}\nReason: {reason}\n")

    def stop(self, message: str, details: dict | None = None) -> None:
        payload = {
            "time": self.now(),
            "run_id": self.run_id,
            "message": message,
            "details": details or {},
            "experiment_root": str(self.experiment_root),
            "output_root": str(self.output_root),
        }
        self.stopped_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stopped_log, "w", encoding="utf-8") as handle:
            handle.write("# STOPPED\n\n")
            handle.write(message + "\n\n")
            handle.write("```json\n")
            json.dump(to_jsonable(payload), handle, indent=2)
            handle.write("\n```\n")
        self.progress("STOPPED", message)
        raise SystemExit(2)

    def mkdirs(self) -> None:
        for path in [
            self.experiment_root,
            self.output_root,
            self.config_root,
            self.checkpoint_root,
            self.run_root,
            self.table_root,
            self.figure_root,
            self.metric_root,
            self.prob_root,
            self.status_root,
            LOGS_ROOT,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def active_pid_is_running(self) -> bool:
        if not self.pid_file.exists():
            return False
        try:
            pid = int(self.pid_file.read_text(encoding="utf-8").strip())
        except ValueError:
            return False
        if pid == os.getpid():
            return False
        try:
            import subprocess

            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def preflight(self) -> None:
        self.mkdirs()
        self.progress("PREFLIGHT", f"start run_id={self.run_id}")
        if self.active_pid_is_running():
            self.stop("Another overnight runner appears to be active.", {"pid_file": str(self.pid_file)})
        self.pid_file.write_text(str(os.getpid()), encoding="utf-8")

        audit_path = OUTPUTS_ROOT / "paper" / "data_audit_report.json"
        if not audit_path.exists():
            self.stop("Data audit report is missing.", {"path": str(audit_path)})
        with open(audit_path, "r", encoding="utf-8") as handle:
            audit = json.load(handle)
        if audit.get("status") != "PASS" or audit.get("issues"):
            self.stop("Data audit report is not PASS.", {"path": str(audit_path), "status": audit.get("status")})

        if not torch.cuda.is_available():
            self.stop("CUDA is not available; full overnight training is not allowed on CPU.")

        free_bytes = shutil.disk_usage(PROJECT_ROOT.anchor).free
        if free_bytes < 10 * 1024**3:
            self.stop("Free disk space is below 10 GB.", {"free_bytes": free_bytes})

        old_weighted_ce = OUTPUTS_ROOT / "paper" / "class_imbalance_study" / "eegnet_weighted_ce"
        old_ckpt = EXPERIMENTS_ROOT / "checkpoints" / "class_imbalance_study" / "eegnet_weighted_ce"
        if old_weighted_ce.exists() or old_ckpt.exists():
            self.decision(
                "Do not reuse the interrupted weighted_ce artifacts automatically.",
                "The old run lacks a complete summary/comparison table and was produced before the final overnight timestamped run contract; "
                "the new run will keep reproducibility clean and avoid overwriting old artifacts.",
            )

        self.decision(
            "Use num_workers=0 by default.",
            "The previous interrupted run failed in a Windows multiprocessing worker while importing Torch with WinError 1455.",
        )
        self.decision(
            "Use timestamped run roots and never delete previous outputs.",
            f"Current run roots are {self.experiment_root} and {self.output_root}.",
        )
        self.write_manifest({"status": "preflight_passed", "audit_report": str(audit_path), "free_bytes": free_bytes})
        self.progress("PREFLIGHT", "complete")

    def write_manifest(self, extra: dict | None = None) -> None:
        payload = {
            "run_id": self.run_id,
            "created_at": self.now(),
            "project_root": str(PROJECT_ROOT),
            "experiment_root": str(self.experiment_root),
            "output_root": str(self.output_root),
            "seed": self.args.seed,
            "epochs": self.args.epochs,
            "batch_size": self.args.batch_size,
            "num_workers": self.args.num_workers,
            "amp": not self.args.no_amp,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "split_paths": {
                "train": str(PROJECT_ROOT / "data/processed/splits/train_metadata.csv"),
                "validation": str(PROJECT_ROOT / "data/processed/splits/validation_metadata.csv"),
                "test": str(PROJECT_ROOT / "data/processed/splits/test_metadata.csv"),
            },
            "env": {key: os.environ.get(key) for key in [
                "TMP", "TEMP", "TMPDIR", "MPLCONFIGDIR", "TORCH_HOME", "XDG_CACHE_HOME",
                "HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "WANDB_DIR",
                "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "MNE_DATA", "TORCHINDUCTOR_CACHE_DIR",
                "CUDA_CACHE_PATH",
            ]},
        }
        if extra:
            payload.update(extra)
        save_json(self.manifest_path, payload)

    def step_deadline(self, fraction_start: float, fraction_end: float) -> float:
        if self.args.smoke:
            return self.started_at + 24 * 3600
        total = self.args.total_hours * 3600
        return self.started_at + total * fraction_end

    def load_metadata(self, smoke: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = load_split_metadata("train")
        val = load_split_metadata("validation")
        test = load_split_metadata("test")
        if smoke:
            train = tiny_balanced(train, 4, self.args.seed + 1)
            val = tiny_balanced(val, 4, self.args.seed + 2)
            test = tiny_balanced(test, 4, self.args.seed + 3)
        return train, val, test

    def run(self) -> None:
        try:
            self.preflight()
            if self.args.smoke:
                self.run_smoke()
            else:
                self.run_full()
            self.cleanup_pid()
        except SystemExit:
            self.cleanup_pid(active_only=True)
            raise
        except Exception as exc:
            tb = traceback.format_exc()
            self.stop("Unhandled runner exception.", {"error": repr(exc), "traceback": tb})

    def cleanup_pid(self, active_only: bool = False) -> None:
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text(encoding="utf-8").strip())
                if pid == os.getpid() or not active_only:
                    self.pid_file.unlink()
            except Exception:
                pass

    def run_smoke(self) -> None:
        self.progress("SMOKE", "start")
        original_epochs = self.args.epochs
        original_batch = self.args.batch_size
        self.args.epochs = 1
        self.args.batch_size = min(self.args.batch_size, 4)
        results = self.train_model_group(
            model_name="eegnet",
            base_strategy=WEIGHTED_CE,
            variants=[WEIGHTED_CE, WEIGHTED_CE_THRESHOLD],
            stage="smoke",
            deadline=time.time() + 24 * 3600,
            smoke=True,
        )
        self.results.extend(results)
        self.args.epochs = original_epochs
        self.args.batch_size = original_batch
        self.generate_publication_assets(final=False)
        self.progress("SMOKE", "complete")

    def run_full(self) -> None:
        step1_deadline = self.step_deadline(0.0, 0.35)
        self.progress("STEP 1", "start EEGNet imbalance study")
        attempted_strategy_ids: set[str] = set()
        for base_strategy, variants in STEP1_GROUPS:
            if time.time() > step1_deadline and self.results:
                self.decision(
                    "Step 1 time budget exceeded; move on with completed imbalance results.",
                    "The time-box rule allows selecting the best strategy from completed attempts under time pressure.",
                )
                break
            group_results = self.attempt_train_group_with_recovery(
                model_name="eegnet",
                base_strategy=base_strategy,
                variants=variants,
                stage="step1_imbalance",
                deadline=step1_deadline,
                smoke=False,
            )
            self.results.extend(group_results)
            attempted_strategy_ids.update(result["strategy_id"] for result in group_results)
            self.write_comparison_tables()

        missing = [s.strategy_id for s in ALL_STEP1_STRATEGIES if s.strategy_id not in attempted_strategy_ids]
        if missing and not self.results:
            self.stop("Step 1 produced no usable imbalance results.", {"missing": missing})
        if missing:
            self.decision(
                "Select imbalance strategy with some Step 1 configurations not completed.",
                f"Missing strategy results due time/failure: {missing}. Selection uses completed actual metrics only.",
            )

        self.best_strategy_result = self.select_best_strategy()
        self.best_strategy = strategy_by_id(self.best_strategy_result["strategy_id"])
        self.progress(
            "STEP 1",
            f"selected {self.best_strategy.strategy_id} val_pr_auc={self.best_strategy_result['validation_metrics']['pr_auc']:.6f} "
            f"val_f1={self.best_strategy_result['validation_metrics']['f1']:.6f}",
        )

        self.progress("STEP 2", "freeze selected imbalance strategy")
        frozen = {
            "strategy": asdict(self.best_strategy),
            "selected_from": self.best_strategy_result,
            "selection_rule": "highest validation PR-AUC, tie-broken by validation F1",
            "frozen_at": self.now(),
        }
        save_json(self.config_root / "frozen_imbalance_strategy.json", frozen)
        self.decision(
            "Freeze imbalance strategy and do not revisit.",
            f"Selected {self.best_strategy.strategy_id} using validation PR-AUC={self.best_strategy_result['validation_metrics']['pr_auc']:.6f} "
            f"and validation F1={self.best_strategy_result['validation_metrics']['f1']:.6f}.",
        )
        self.progress("STEP 2", "complete")

        self.progress("STEP 3", "start final model training")
        step3_deadline = self.step_deadline(0.35, 0.85)
        for model_name in ["eegnet", "attention", "hybrid"]:
            if time.time() > step3_deadline and model_name == "hybrid":
                self.decision(
                    "Skip Hybrid final training due Step 3 time pressure.",
                    "The time-box rule prioritizes completing EEGNet and Attention fully before Hybrid.",
                )
                self.write_status(
                    "step3_final",
                    model_name,
                    self.best_strategy.strategy_id,
                    {"status": "INCOMPLETE", "reason": "Step 3 time budget exceeded before Hybrid start"},
                )
                continue
            final_results = self.attempt_train_group_with_recovery(
                model_name=model_name,
                base_strategy=self.best_strategy,
                variants=[self.best_strategy],
                stage="step3_final",
                deadline=step3_deadline,
                smoke=False,
            )
            self.results.extend(final_results)
            self.write_comparison_tables()
        self.progress("STEP 3", "complete")

        self.progress("STEP 4", "final evaluation assets generated during training/evaluation")
        self.write_comparison_tables()
        self.progress("STEP 4", "complete")

        self.progress("STEP 5", "start publication assets")
        self.generate_publication_assets(final=True)
        self.progress("STEP 5", "complete")

    def attempt_train_group_with_recovery(
        self,
        model_name: str,
        base_strategy: Strategy,
        variants: list[Strategy],
        stage: str,
        deadline: float,
        smoke: bool,
    ) -> list[dict]:
        try:
            return self.train_model_group(model_name, base_strategy, variants, stage, deadline, smoke)
        except RuntimeError as exc:
            if is_oom(exc) and self.args.batch_size > 64:
                self.decision(
                    "Retry failed run once with smaller batch size.",
                    f"{model_name}/{base_strategy.base_id} failed with CUDA OOM at batch_size={self.args.batch_size}; retrying with 64.",
                )
                old_batch = self.args.batch_size
                self.args.batch_size = 64
                try:
                    return self.train_model_group(model_name, base_strategy, variants, stage, deadline, smoke)
                except Exception as retry_exc:
                    self.args.batch_size = old_batch
                    return self.mark_group_failed(model_name, base_strategy, variants, stage, retry_exc)
                finally:
                    self.args.batch_size = old_batch
            return self.mark_group_failed(model_name, base_strategy, variants, stage, exc)
        except Exception as exc:
            return self.mark_group_failed(model_name, base_strategy, variants, stage, exc)

    def mark_group_failed(
        self,
        model_name: str,
        base_strategy: Strategy,
        variants: list[Strategy],
        stage: str,
        exc: Exception,
    ) -> list[dict]:
        self.progress(stage.upper(), f"FAILED {model_name}/{base_strategy.base_id}: {repr(exc)}")
        self.decision(
            "Skip failed configuration and continue.",
            f"{model_name}/{base_strategy.base_id} failed with {repr(exc)}. Traceback saved in status JSON.",
        )
        failure = {
            "status": "FAILED",
            "model": model_name,
            "base_strategy": base_strategy.base_id,
            "variants": [v.strategy_id for v in variants],
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        self.write_status(stage, model_name, base_strategy.base_id, failure)
        return [
            {
                "status": "FAILED",
                "model": model_name,
                "strategy_id": variant.strategy_id,
                "display_name": variant.display_name,
                "base_strategy": base_strategy.base_id,
                "error": repr(exc),
            }
            for variant in variants
        ]

    def train_model_group(
        self,
        model_name: str,
        base_strategy: Strategy,
        variants: list[Strategy],
        stage: str,
        deadline: float,
        smoke: bool,
    ) -> list[dict]:
        run_key = f"{stage}_{model_name}_{base_strategy.base_id}"
        run_dir = self.run_root / run_key
        ckpt_dir = self.checkpoint_root / run_key
        metric_dir = self.metric_root / run_key
        prob_dir = self.prob_root / run_key
        fig_dir = self.figure_root / run_key
        for path in [run_dir, ckpt_dir, metric_dir, prob_dir, fig_dir]:
            path.mkdir(parents=True, exist_ok=True)

        self.progress(stage.upper(), f"START {model_name}/{base_strategy.base_id}")
        set_deterministic_seed(self.args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda")
        use_amp = (not self.args.no_amp) and device.type == "cuda"

        train_meta, val_meta, test_meta = self.load_metadata(smoke=smoke)
        train_loader, val_loader, test_loader = self.create_loaders(train_meta, val_meta, test_meta, base_strategy)
        model = build_model(model_name).to(device)
        class_weights = (
            compute_class_weights_from_labels(train_loader.dataset._labels)
            if base_strategy.class_weights == "auto"
            else None
        )
        criterion = build_loss(
            base_strategy.loss_name,
            class_weights=class_weights,
            focal_gamma=self.args.focal_gamma,
            device=device,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.args.scheduler_factor,
            patience=self.args.scheduler_patience,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        config = {
            "stage": stage,
            "model": model_name,
            "base_strategy": asdict(base_strategy),
            "variants": [asdict(v) for v in variants],
            "epochs": self.args.epochs,
            "batch_size": self.args.batch_size,
            "num_workers": self.args.num_workers,
            "lr": self.args.lr,
            "weight_decay": self.args.weight_decay,
            "focal_gamma": self.args.focal_gamma,
            "seed": self.args.seed,
            "amp": use_amp,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
        }
        save_json(run_dir / "config.json", config)

        last_path = ckpt_dir / "last.pth"
        start_epoch = 0
        best_scores = {variant.strategy_id: [-float("inf"), -float("inf")] for variant in variants}
        best_thresholds = {variant.strategy_id: 0.5 for variant in variants}
        history: list[dict] = []
        bad_epochs = 0
        if self.args.resume and last_path.exists():
            checkpoint = load_checkpoint(last_path, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, device=device)
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            history = checkpoint.get("history", [])
            best_scores = checkpoint.get("best_scores", best_scores)
            best_thresholds = checkpoint.get("best_thresholds", best_thresholds)
            bad_epochs = int(checkpoint.get("bad_epochs", 0))
            self.progress(stage.upper(), f"RESUME {model_name}/{base_strategy.base_id} epoch={start_epoch}")

        started = time.time()
        completed_training = False
        for epoch in range(start_epoch, self.args.epochs):
            epoch_start = time.time()
            model.train()
            running_loss = 0.0
            batches = 0
            for xb, yb in train_loader:
                xb, yb = move_batch_to_device(xb, yb, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                if self.args.gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.detach().cpu())
                batches += 1
            train_loss = running_loss / max(batches, 1)

            val_true, val_prob, val_seconds = self.collect_probabilities(model, val_loader, device, use_amp)
            epoch_rows = []
            improved = False
            scheduler_score = -float("inf")
            for variant in variants:
                threshold, val_metrics = evaluate_variant(variant, val_true, val_prob)
                selection = [float(val_metrics["pr_auc"]), float(val_metrics["f1"])]
                scheduler_score = max(scheduler_score, selection[0] + selection[1])
                row = {
                    "epoch": epoch + 1,
                    "model": model_name,
                    "stage": stage,
                    "base_strategy": base_strategy.base_id,
                    "strategy_id": variant.strategy_id,
                    "threshold": threshold,
                    "train_loss": train_loss,
                    "epoch_seconds": round(time.time() - epoch_start, 3),
                    "validation_seconds": round(val_seconds, 3),
                    **{f"val_{metric}": val_metrics[metric] for metric in METRIC_COLUMNS},
                }
                epoch_rows.append(row)
                if tuple(selection) > tuple(best_scores.get(variant.strategy_id, [-float("inf"), -float("inf")])):
                    improved = True
                    best_scores[variant.strategy_id] = selection
                    best_thresholds[variant.strategy_id] = threshold
                    state = checkpoint_state(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        selection[0] + selection[1],
                        threshold,
                        {**config, "strategy_id": variant.strategy_id},
                    )
                    state.update(
                        {
                            "best_scores": best_scores,
                            "best_thresholds": best_thresholds,
                            "history": history + epoch_rows,
                            "bad_epochs": 0,
                        }
                    )
                    torch.save(state, ckpt_dir / f"{variant.strategy_id}_best.pth")

            scheduler.step(scheduler_score)
            bad_epochs = 0 if improved else bad_epochs + 1
            history.extend(epoch_rows)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            last_state = checkpoint_state(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                scheduler_score,
                0.5,
                config,
            )
            last_state.update(
                {
                    "best_scores": best_scores,
                    "best_thresholds": best_thresholds,
                    "history": history,
                    "bad_epochs": bad_epochs,
                }
            )
            torch.save(last_state, last_path)
            msg = " | ".join(
                f"{row['strategy_id']} val_pr_auc={row['val_pr_auc']:.4f} val_f1={row['val_f1']:.4f} thr={row['threshold']:.2f}"
                for row in epoch_rows
            )
            self.progress(stage.upper(), f"EPOCH {model_name}/{base_strategy.base_id} {epoch + 1}/{self.args.epochs} loss={train_loss:.4f} {msg}")
            if bad_epochs >= self.args.early_stopping_patience:
                self.progress(stage.upper(), f"EARLY_STOP {model_name}/{base_strategy.base_id} epoch={epoch + 1}")
                completed_training = True
                break
            if time.time() > deadline and not smoke:
                self.progress(stage.upper(), f"TIMEBOX_REACHED {model_name}/{base_strategy.base_id} after epoch={epoch + 1}")
                break
        else:
            completed_training = True

        if not last_path.exists():
            raise RuntimeError(f"Missing last checkpoint for {run_key}")

        results: list[dict] = []
        history_df = pd.DataFrame(history)
        peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        for variant in variants:
            best_path = ckpt_dir / f"{variant.strategy_id}_best.pth"
            if not best_path.exists():
                self.write_status(stage, model_name, variant.strategy_id, {"status": "FAILED", "reason": "missing best checkpoint"})
                continue
            eval_model = build_model(model_name).to(device)
            checkpoint = load_checkpoint(best_path, eval_model, device=device)
            threshold = float(checkpoint.get("best_threshold", best_thresholds.get(variant.strategy_id, 0.5)))
            val_true, val_prob, val_seconds = self.collect_probabilities(eval_model, val_loader, device, use_amp)
            test_true, test_prob, test_seconds = self.collect_probabilities(eval_model, test_loader, device, use_amp)
            val_metrics = binary_classification_metrics(val_true, val_prob, threshold)
            test_metrics = binary_classification_metrics(test_true, test_prob, threshold)
            threshold_df = threshold_sweep(val_true, val_prob)
            metric_payload = {
                "status": "COMPLETED" if completed_training else "INCOMPLETE",
                "model": model_name,
                "stage": stage,
                "strategy_id": variant.strategy_id,
                "display_name": variant.display_name,
                "base_strategy": base_strategy.base_id,
                "checkpoint_best": str(best_path),
                "checkpoint_last": str(last_path),
                "threshold": threshold,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "training_seconds": round(time.time() - started, 3),
                "validation_inference_seconds": round(val_seconds, 3),
                "test_inference_seconds": round(test_seconds, 3),
                "test_samples": int(len(test_true)),
                "inference_samples_per_second": float(len(test_true) / test_seconds) if test_seconds > 0 else None,
                "peak_cuda_memory_bytes": peak_memory,
                "parameter_count": count_parameters(eval_model),
                "history_csv": str(run_dir / "history.csv"),
                "metric_json": str(metric_dir / f"{variant.strategy_id}_metrics.json"),
            }
            save_json(metric_dir / f"{variant.strategy_id}_metrics.json", metric_payload)
            np.savez_compressed(
                prob_dir / f"{variant.strategy_id}_probabilities.npz",
                val_true=val_true,
                val_prob=val_prob,
                test_true=test_true,
                test_prob=test_prob,
                threshold=np.asarray([threshold], dtype=np.float32),
            )
            threshold_df.to_csv(metric_dir / f"{variant.strategy_id}_threshold_sweep.csv", index=False)
            self.plot_all_artifacts(
                fig_dir / variant.strategy_id,
                f"{model_name} - {variant.display_name}",
                test_true,
                test_prob,
                val_true,
                val_prob,
                threshold,
                history_df[history_df["strategy_id"] == variant.strategy_id].copy(),
                threshold_df,
            )
            self.write_status(stage, model_name, variant.strategy_id, metric_payload)
            results.append(metric_payload)
            self.training_records.append(metric_payload)

        self.progress(stage.upper(), f"END {model_name}/{base_strategy.base_id} results={len(results)}")
        return results

    def create_loaders(
        self,
        train_meta: pd.DataFrame,
        val_meta: pd.DataFrame,
        test_meta: pd.DataFrame,
        strategy: Strategy,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        shards = discover_shards()
        train_dataset = ShardedEEGDataset(train_meta, shards=shards)
        val_dataset = ShardedEEGDataset(val_meta, shards=shards)
        test_dataset = ShardedEEGDataset(test_meta, shards=shards)
        common = {
            "num_workers": self.args.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": self.args.num_workers > 0,
        }
        if self.args.num_workers > 0:
            common["prefetch_factor"] = self.args.prefetch_factor
        if strategy.sampler == "weighted_random":
            labels = train_dataset._labels
            counts = np.bincount(labels, minlength=2).astype(np.float64)
            weights = 1.0 / counts[labels]
            generator = torch.Generator()
            generator.manual_seed(self.args.seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                sampler=WeightedRandomSampler(
                    weights=torch.as_tensor(weights, dtype=torch.double),
                    num_samples=len(weights),
                    replacement=True,
                    generator=generator,
                ),
                **common,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=ShardBatchSampler(
                    train_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    seed=self.args.seed,
                ),
                **common,
            )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=ShardBatchSampler(val_dataset, self.args.batch_size, False, seed=self.args.seed),
            **common,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=ShardBatchSampler(test_dataset, self.args.batch_size, False, seed=self.args.seed),
            **common,
        )
        return train_loader, val_loader, test_loader

    def collect_probabilities(self, model, loader, device, use_amp: bool) -> tuple[np.ndarray, np.ndarray, float]:
        model.eval()
        labels = []
        probs = []
        start = time.time()
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = move_batch_to_device(xb, yb, device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(xb)
                    batch_prob = torch.softmax(logits, dim=1)[:, 1]
                labels.append(yb.detach().cpu().numpy())
                probs.append(batch_prob.detach().cpu().numpy())
        seconds = time.time() - start
        return np.concatenate(labels), np.concatenate(probs), seconds

    def plot_all_artifacts(
        self,
        fig_dir: Path,
        title: str,
        test_true: np.ndarray,
        test_prob: np.ndarray,
        val_true: np.ndarray,
        val_prob: np.ndarray,
        threshold: float,
        history: pd.DataFrame,
        threshold_df: pd.DataFrame,
    ) -> None:
        fig_dir.mkdir(parents=True, exist_ok=True)
        y_pred = (test_prob >= threshold).astype(np.int64)
        cm = confusion_matrix(test_true, y_pred, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["interictal", "preictal"]).plot(values_format="d")
        plt.title(f"{title} Confusion Matrix")
        save_figure(fig_dir / "confusion_matrix.png")

        fpr, tpr, _ = roc_curve(test_true, test_prob)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_score(test_true, test_prob)).plot()
        plt.title(f"{title} ROC")
        save_figure(fig_dir / "roc_curve.png")

        precision, recall, _ = precision_recall_curve(test_true, test_prob)
        PrecisionRecallDisplay(
            precision=precision,
            recall=recall,
            average_precision=average_precision_score(test_true, test_prob),
        ).plot()
        plt.title(f"{title} Precision-Recall")
        save_figure(fig_dir / "precision_recall_curve.png")

        if len(history):
            plt.figure(figsize=(9, 5))
            plt.plot(history["epoch"], history["train_loss"], label="train_loss")
            for column in ["val_pr_auc", "val_f1", "val_balanced_accuracy"]:
                if column in history:
                    plt.plot(history["epoch"], history[column], label=column)
            plt.xlabel("epoch")
            plt.legend()
            plt.title(f"{title} Learning Curves")
            save_figure(fig_dir / "learning_curves.png")

            plt.figure(figsize=(9, 5))
            for column in ["val_accuracy", "val_precision", "val_recall", "val_f1", "val_pr_auc", "val_roc_auc"]:
                if column in history:
                    plt.plot(history["epoch"], history[column], label=column)
            plt.xlabel("epoch")
            plt.legend()
            plt.title(f"{title} Validation Curves")
            save_figure(fig_dir / "validation_curves.png")

        plt.figure(figsize=(8, 5))
        plt.plot(threshold_df["threshold"], threshold_df["f1"], label="f1")
        plt.plot(threshold_df["threshold"], threshold_df["balanced_accuracy"], label="balanced_accuracy")
        plt.plot(threshold_df["threshold"], threshold_df["precision"], label="precision")
        plt.plot(threshold_df["threshold"], threshold_df["recall"], label="recall")
        plt.axvline(threshold, color="black", linestyle="--", label="selected")
        plt.xlabel("threshold")
        plt.legend()
        plt.title(f"{title} Threshold Curves")
        save_figure(fig_dir / "threshold_curves.png")

        prob_true, prob_pred = calibration_curve(test_true, test_prob, n_bins=10, strategy="uniform")
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
        plt.plot(prob_pred, prob_true, marker="o", label="model")
        plt.xlabel("mean predicted probability")
        plt.ylabel("fraction positives")
        plt.legend()
        plt.title(f"{title} Calibration Curve")
        save_figure(fig_dir / "calibration_curve.png")

    def write_status(self, stage: str, model_name: str, strategy_id: str, payload: dict) -> None:
        path = self.status_root / f"{stage}_{model_name}_{strategy_id}.json"
        save_json(path, payload)

    def select_best_strategy(self) -> dict:
        completed = [
            result for result in self.results
            if result.get("model") == "eegnet"
            and result.get("stage") == "step1_imbalance"
            and result.get("status") in {"COMPLETED", "INCOMPLETE"}
            and "validation_metrics" in result
        ]
        if not completed:
            self.stop("No completed Step 1 EEGNet strategy is available for selection.")
        completed.sort(
            key=lambda result: (
                float(result["validation_metrics"]["pr_auc"]),
                float(result["validation_metrics"]["f1"]),
            ),
            reverse=True,
        )
        return completed[0]

    def write_comparison_tables(self) -> None:
        if not self.results:
            return
        rows = []
        for result in self.results:
            row = {
                "status": result.get("status"),
                "stage": result.get("stage"),
                "model": result.get("model"),
                "strategy_id": result.get("strategy_id"),
                "display_name": result.get("display_name"),
                "base_strategy": result.get("base_strategy"),
                "threshold": result.get("threshold"),
                "checkpoint_best": result.get("checkpoint_best"),
                "checkpoint_last": result.get("checkpoint_last"),
                "metric_json": result.get("metric_json"),
            }
            for prefix in ["validation", "test"]:
                metrics = result.get(f"{prefix}_metrics", {})
                for metric in METRIC_COLUMNS:
                    row[f"{prefix}_{metric}"] = metrics.get(metric)
            rows.append(row)
        table = pd.DataFrame(rows)
        self.table_root.mkdir(parents=True, exist_ok=True)
        table.to_csv(self.table_root / "model_comparison.csv", index=False)
        write_latex(table, self.table_root / "model_comparison.tex")

    def generate_publication_assets(self, final: bool) -> None:
        self.table_root.mkdir(parents=True, exist_ok=True)
        audit_path = OUTPUTS_ROOT / "paper" / "data_audit_report.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}

        dataset_summary = pd.DataFrame([
            {
                "total_windows": audit.get("total_windows"),
                "total_patients": audit.get("total_patients"),
                "total_edfs": audit.get("total_edfs"),
                "total_seizure_events": audit.get("total_seizure_events"),
                **{f"overall_{k}": v for k, v in audit.get("overall_class_distribution", {}).items()},
            }
        ])
        dataset_summary.to_csv(self.table_root / "dataset_summary.csv", index=False)
        write_latex(dataset_summary, self.table_root / "dataset_summary.tex")

        patient_split = pd.DataFrame(audit.get("patient_distribution_by_split", []))
        if len(patient_split):
            patient_split.to_csv(self.table_root / "patient_split_summary.csv", index=False)
            write_latex(patient_split, self.table_root / "patient_split_summary.tex")

        hyperparams = pd.DataFrame([
            {
                "seed": self.args.seed,
                "epochs": self.args.epochs,
                "batch_size": self.args.batch_size,
                "num_workers": self.args.num_workers,
                "lr": self.args.lr,
                "weight_decay": self.args.weight_decay,
                "focal_gamma": self.args.focal_gamma,
                "optimizer": "AdamW",
                "scheduler": "ReduceLROnPlateau",
                "amp": not self.args.no_amp,
            }
        ])
        hyperparams.to_csv(self.table_root / "hyperparameter_summary.csv", index=False)
        write_latex(hyperparams, self.table_root / "hyperparameter_summary.tex")
        hyperparams.to_csv(self.table_root / "training_configuration.csv", index=False)
        write_latex(hyperparams, self.table_root / "training_configuration.tex")

        completed = [r for r in self.results if r.get("status") in {"COMPLETED", "INCOMPLETE"} and "test_metrics" in r]
        if completed:
            final_rows = []
            for result in completed:
                row = {
                    "stage": result.get("stage"),
                    "model": result.get("model"),
                    "strategy_id": result.get("strategy_id"),
                    "status": result.get("status"),
                    "threshold": result.get("threshold"),
                }
                for metric in METRIC_COLUMNS:
                    row[metric] = result["test_metrics"].get(metric)
                final_rows.append(row)
            final_metrics = pd.DataFrame(final_rows)
            final_metrics.to_csv(self.table_root / "final_metrics.csv", index=False)
            write_latex(final_metrics, self.table_root / "final_metrics.tex")

            parameter_counts = pd.DataFrame([
                {
                    "stage": r.get("stage"),
                    "model": r.get("model"),
                    "strategy_id": r.get("strategy_id"),
                    "parameter_count": r.get("parameter_count"),
                }
                for r in completed
            ])
            parameter_counts.to_csv(self.table_root / "parameter_count.csv", index=False)
            write_latex(parameter_counts, self.table_root / "parameter_count.tex")

            training_time = pd.DataFrame([
                {
                    "stage": r.get("stage"),
                    "model": r.get("model"),
                    "strategy_id": r.get("strategy_id"),
                    "training_seconds": r.get("training_seconds"),
                }
                for r in completed
            ])
            training_time.to_csv(self.table_root / "training_time.csv", index=False)
            write_latex(training_time, self.table_root / "training_time.tex")

            gpu_memory = pd.DataFrame([
                {
                    "stage": r.get("stage"),
                    "model": r.get("model"),
                    "strategy_id": r.get("strategy_id"),
                    "peak_cuda_memory_bytes": r.get("peak_cuda_memory_bytes"),
                }
                for r in completed
            ])
            gpu_memory.to_csv(self.table_root / "gpu_memory.csv", index=False)
            write_latex(gpu_memory, self.table_root / "gpu_memory.tex")

            inference_speed = pd.DataFrame([
                {
                    "stage": r.get("stage"),
                    "model": r.get("model"),
                    "strategy_id": r.get("strategy_id"),
                    "test_samples": r.get("test_samples"),
                    "test_inference_seconds": r.get("test_inference_seconds"),
                    "inference_samples_per_second": r.get("inference_samples_per_second"),
                }
                for r in completed
            ])
            inference_speed.to_csv(self.table_root / "inference_speed.csv", index=False)
            write_latex(inference_speed, self.table_root / "inference_speed.tex")

        self.write_comparison_tables()
        self.write_final_report(final=final)

    def write_final_report(self, final: bool) -> None:
        report_path = self.output_root / "FINAL_REPORT.md"
        completed = [r for r in self.results if r.get("status") in {"COMPLETED", "INCOMPLETE"} and "test_metrics" in r]
        failed = [r for r in self.results if r.get("status") == "FAILED"]
        lines = [
            "# Overnight Experimental Report",
            "",
            f"Run ID: `{self.run_id}`",
            f"Generated: `{self.now()}`",
            f"Final: `{final}`",
            "",
            "## Imbalance Strategy",
        ]
        if self.best_strategy_result:
            vm = self.best_strategy_result["validation_metrics"]
            lines.extend([
                f"Selected: `{self.best_strategy_result['strategy_id']}`",
                f"Validation PR-AUC: `{vm['pr_auc']}`",
                f"Validation F1: `{vm['f1']}`",
            ])
        else:
            lines.append("Not selected yet.")
        lines.extend(["", "## Completed / Incomplete Runs", ""])
        for result in completed:
            tm = result["test_metrics"]
            lines.append(
                f"- `{result.get('stage')}` `{result.get('model')}` `{result.get('strategy_id')}` "
                f"status=`{result.get('status')}` test_pr_auc=`{tm.get('pr_auc')}` test_f1=`{tm.get('f1')}`"
            )
        lines.extend(["", "## Failed Runs", ""])
        if failed:
            for result in failed:
                lines.append(f"- `{result.get('model')}` `{result.get('strategy_id')}`: {result.get('error')}")
        else:
            lines.append("None recorded.")
        lines.extend([
            "",
            "## Autonomous Decisions",
            "",
            f"See `{self.decision_log}`.",
            "",
            "## Limitations",
            "",
            "- Metrics are reported only for actual completed evaluations.",
            "- No placeholders or interpolated results were generated.",
            "",
            "## Recommendations",
            "",
            "- Use the frozen imbalance strategy for paper model comparisons.",
            "- Report incomplete or failed configurations transparently if any remain.",
        ])
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        final_entry = OUTPUTS_ROOT / "paper" / "FINAL_REPORT.md"
        if final and not final_entry.exists():
            shutil.copy2(report_path, final_entry)
            self.decision("Copy final report to outputs/paper/FINAL_REPORT.md.", "The entrypoint did not already exist.")
        elif final and final_entry.exists():
            self.decision(
                "Do not overwrite outputs/paper/FINAL_REPORT.md.",
                f"Existing final report preserved; new report is {report_path}.",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--total-hours", type=float, default=8.0)
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        output = float(value)
        return None if not math.isfinite(output) else output
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2)


def write_latex(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(df.to_latex(index=False, escape=True))


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def tiny_balanced(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    parts = []
    for label in [0, 1]:
        label_df = df[df["label"].astype(int) == label]
        if label_df.empty:
            raise RuntimeError(f"Cannot smoke-test split with no label {label}")
        parts.append(label_df.sample(n=min(per_class, len(label_df)), random_state=seed + label))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def is_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda" in message and "memory" in message


def evaluate_variant(strategy: Strategy, y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict]:
    if strategy.threshold_mode == "optimized":
        threshold, metrics = optimize_threshold(y_true, y_prob, metric="f1")
    else:
        threshold = 0.5
        metrics = binary_classification_metrics(y_true, y_prob, threshold)
    return float(threshold), metrics


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        metrics = binary_classification_metrics(y_true, y_prob, float(threshold))
        rows.append({key: metrics[key] for key in ["threshold", *METRIC_COLUMNS, "tn", "fp", "fn", "tp"]})
    return pd.DataFrame(rows)


def count_parameters(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def strategy_by_id(strategy_id: str) -> Strategy:
    for strategy in ALL_STEP1_STRATEGIES:
        if strategy.strategy_id == strategy_id:
            return strategy
    raise KeyError(strategy_id)


def main() -> None:
    args = parse_args()
    runner = OvernightRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
