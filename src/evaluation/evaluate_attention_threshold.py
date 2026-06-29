import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.project_paths import ensure_project_storage, experiment_path, output_path, print_storage_summary, project_path

ensure_project_storage()

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
)

from src.models.hybrid_attention import HybridEEGNetAttention


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


def main():
    fig_path = output_path("figures/attention_confusion_matrix_final.png")
    result_path = experiment_path("results/attention_final_metrics.txt")
    print_storage_summary({"figure": fig_path, "metrics": result_path})

    if len(sys.argv) > 1:
        threshold = float(sys.argv[1])
    else:
        threshold_file = experiment_path("results/attention_best_threshold.txt")
        threshold = float(threshold_file.read_text().strip())

    print(f"Using threshold: {threshold:.2f}")

    X = np.load(project_path("data/processed/X.npy"))
    y = np.load(project_path("data/processed/y.npy"))

    test_meta = pd.read_csv(project_path("data/processed/splits/test_metadata.csv"))
    test_idx = test_meta["dataset_index"].values

    X_test = X[test_idx]
    y_test = y[test_idx]

    device = torch.device("cpu")
    model = HybridEEGNetAttention().to(device)
    model.load_state_dict(
        torch.load(experiment_path("checkpoints/attention_best.pth"), map_location=device)
    )

    test_loader = make_loader(X_test, y_test)
    y_prob, y_true = collect_probs(model, test_loader, device)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n================================")
    print("ATTENTION MODEL FINAL RESULTS")
    print("================================")
    print("Accuracy :", round(acc, 4))
    print("F1 Score :", round(f1, 4))
    print("Balanced Accuracy :", round(bal_acc, 4))
    print("Precision :", round(prec, 4))
    print("Recall :", round(rec, 4))

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(classification_report(y_true, y_pred, zero_division=0))

    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Attention Confusion Matrix (thr={threshold:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Interictal", "Preictal"])
    plt.yticks([0, 1], ["Interictal", "Preictal"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved figure: {fig_path}")

    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"threshold={threshold:.2f}\n")
        f.write(f"accuracy={acc:.4f}\n")
        f.write(f"f1={f1:.4f}\n")
        f.write(f"balanced_accuracy={bal_acc:.4f}\n")
        f.write(f"precision={prec:.4f}\n")
        f.write(f"recall={rec:.4f}\n")
        f.write(f"cm={cm.tolist()}\n")

    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
