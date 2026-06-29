import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.models.hybrid_attention import HybridEEGNetAttention
from src.training.sharded_training import (
    create_sharded_loaders,
    get_device,
    move_batch_to_device,
)
from src.utils.project_paths import ensure_project_storage, experiment_path, print_storage_summary


def evaluate(model, loader, device):
    model.eval()
    preds = []
    truth = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = move_batch_to_device(xb, yb, device)
            outputs = model(xb)
            pred = torch.argmax(outputs, dim=1)
            preds.extend(pred.cpu().numpy())
            truth.extend(yb.cpu().numpy())
    return truth, preds


def main():
    ensure_project_storage()
    checkpoint_path = experiment_path("checkpoints/attention_best.pth")
    print_storage_summary({"checkpoint": checkpoint_path})
    print("Loading sharded dataset...")
    train_loader, val_loader, test_loader, _ = create_sharded_loaders(batch_size=64)

    device = get_device()
    print("Device:", device)

    model = HybridEEGNetAttention().to(device)
    class_weights = torch.tensor([1.0, 1.1], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    epochs = 25
    best_score = -1.0
    patience = 5
    bad_epochs = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nTraining Started...\n")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = move_batch_to_device(xb, yb, device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        truth, preds = evaluate(model, val_loader, device)
        val_acc = accuracy_score(truth, preds)
        val_f1 = f1_score(truth, preds, zero_division=0)
        val_bal_acc = balanced_accuracy_score(truth, preds)
        score = (val_f1 + val_bal_acc) / 2

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | Loss={train_loss:.4f} | "
            f"Val Acc={val_acc:.4f} | Val F1={val_f1:.4f} | "
            f"Val BalAcc={val_bal_acc:.4f} | Score={score:.4f}"
        )

        scheduler.step(score)
        if score > best_score:
            best_score = score
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New Best Model Saved (Score={best_score:.4f})")
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("Early stopping triggered.")
            break

    print("\nLoading Best Model...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print("\nRunning Test Evaluation...\n")
    truth, preds = evaluate(model, test_loader, device)
    test_acc = accuracy_score(truth, preds)
    test_f1 = f1_score(truth, preds, zero_division=0)
    test_bal_acc = balanced_accuracy_score(truth, preds)

    print("\n================================")
    print("ATTENTION MODEL RESULTS")
    print("================================")
    print("Accuracy :", round(test_acc, 4))
    print("F1 Score :", round(test_f1, 4))
    print("Balanced Accuracy :", round(test_bal_acc, 4))
    print("\nConfusion Matrix")
    print(confusion_matrix(truth, preds))
    print("\nClassification Report")
    print(classification_report(truth, preds, zero_division=0))
    print("\nBest Validation Score")
    print(round(best_score, 4))
    print("\nSaved Model:")
    print(checkpoint_path)


if __name__ == "__main__":
    main()
