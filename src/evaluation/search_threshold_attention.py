import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score
)

from src.models.hybrid_attention import HybridEEGNetAttention
from src.utils.project_paths import ensure_project_storage, experiment_path, print_storage_summary, project_path


def make_loader(X, y, batch_size=64):
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)


def collect_probs(model, loader, device):
    model.eval()
    probs = []
    truth = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(p.cpu().numpy())
            truth.extend(yb.numpy())

    return np.array(probs), np.array(truth)


def metrics_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    score = 0.5 * f1 + 0.5 * bal_acc

    return {
        "threshold": thr,
        "accuracy": acc,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "score": score,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
    }


def main():
    ensure_project_storage()
    out_csv = experiment_path("results/threshold_search_attention.csv")
    threshold_path = experiment_path("results/attention_best_threshold.txt")
    print_storage_summary({"threshold_csv": out_csv, "threshold_file": threshold_path})

    X = np.load(project_path("data/processed/X.npy"))
    y = np.load(project_path("data/processed/y.npy"))

    val_meta = pd.read_csv(project_path("data/processed/splits/val_metadata.csv"))
    val_idx = val_meta["dataset_index"].values

    X_val = X[val_idx]
    y_val = y[val_idx]

    device = torch.device("cpu")
    model = HybridEEGNetAttention().to(device)
    model.load_state_dict(
        torch.load(experiment_path("checkpoints/attention_best.pth"), map_location=device)
    )

    val_loader = make_loader(X_val, y_val, batch_size=64)
    val_prob, val_true = collect_probs(model, val_loader, device)

    rows = []
    thresholds = np.arange(0.01, 0.51, 0.01)

    for thr in thresholds:
        rows.append(metrics_at_threshold(val_true, val_prob, float(round(thr, 2))))

    df = pd.DataFrame(rows).sort_values(
        by=["score", "balanced_accuracy", "f1"],
        ascending=False
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    best = df.iloc[0]

    print("\nTOP 10 THRESHOLDS")
    print(df.head(10)[[
        "threshold", "accuracy", "f1", "balanced_accuracy",
        "precision", "recall", "score", "tn", "fp", "fn", "tp"
    ]])

    print("\nBEST THRESHOLD")
    print(f"threshold = {best['threshold']:.2f}")
    print(f"score     = {best['score']:.4f}")
    print(f"acc       = {best['accuracy']:.4f}")
    print(f"f1        = {best['f1']:.4f}")
    print(f"bal_acc   = {best['balanced_accuracy']:.4f}")
    print(f"precision = {best['precision']:.4f}")
    print(f"recall    = {best['recall']:.4f}")
    print(f"cm        = [[{int(best['tn'])}, {int(best['fp'])}], [{int(best['fn'])}, {int(best['tp'])}]]")

    with open(threshold_path, "w", encoding="utf-8") as f:
        f.write(f"{best['threshold']:.2f}")

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {threshold_path}")


if __name__ == "__main__":
    main()
