from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights_from_labels(
    labels,
    num_classes: int = 2,
    normalize: bool = True,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Cannot compute class weights with empty class counts: {counts}")

    weights = labels.size / (num_classes * counts)
    if normalize:
        weights = weights / weights.mean()
    return weights.astype(np.float32)


def compute_class_weights_from_split(
    split_metadata_path: str | Path,
    num_classes: int = 2,
) -> np.ndarray:
    split = pd.read_csv(split_metadata_path)
    if "label" not in split.columns:
        raise ValueError(f"{split_metadata_path} does not contain a label column")
    return compute_class_weights_from_labels(split["label"].to_numpy(), num_classes)


def build_loss(
    loss_name: str,
    class_weights=None,
    focal_gamma: float = 2.0,
    device: torch.device | str = "cpu",
) -> nn.Module:
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    loss_name = loss_name.lower()
    if loss_name in {"ce", "cross_entropy", "weighted_ce", "weighted_cross_entropy"}:
        return nn.CrossEntropyLoss(weight=weight_tensor)
    if loss_name in {"focal", "focal_loss"}:
        return FocalLoss(alpha=weight_tensor, gamma=focal_gamma)

    raise ValueError(f"Unknown loss function: {loss_name}")


def binary_classification_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> dict[str, float | int | list[list[int]]]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
    }


def optimize_threshold(
    y_true,
    y_prob,
    thresholds=None,
    metric: str = "f1",
) -> tuple[float, dict]:
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

    best_threshold = None
    best_metrics = None
    best_score = -np.inf

    for threshold in thresholds:
        metrics = binary_classification_metrics(y_true, y_prob, float(threshold))
        if metric == "score":
            score = 0.5 * metrics["f1"] + 0.5 * metrics["balanced_accuracy"]
        else:
            if metric not in metrics:
                raise ValueError(f"Unknown threshold metric: {metric}")
            score = metrics[metric]

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics
