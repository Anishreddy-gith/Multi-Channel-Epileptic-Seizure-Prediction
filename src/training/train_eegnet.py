import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.training.sharded_training import (
    create_sharded_loaders,
    get_device,
    move_batch_to_device,
)
from src.utils.project_paths import ensure_project_storage, experiment_path, print_storage_summary


class EEGNet(nn.Module):
    def __init__(self):
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
        x = self.features(x)
        return self.classifier(x)


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
    checkpoint_path = experiment_path("checkpoints/eegnet_baseline.pth")
    print_storage_summary({"checkpoint": checkpoint_path})
    print("Loading sharded dataset...")
    train_loader, val_loader, test_loader, _ = create_sharded_loaders(batch_size=32)

    device = get_device()
    print("Device:", device)

    model = EEGNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 15

    print("\nStarting Training...\n")
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
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Loss={train_loss:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f}"
        )

    print("\nRunning Test Evaluation...\n")
    truth, preds = evaluate(model, test_loader, device)
    test_acc = accuracy_score(truth, preds)
    test_f1 = f1_score(truth, preds, zero_division=0)

    print("================================")
    print("FINAL TEST RESULTS")
    print("================================")
    print("Accuracy :", round(test_acc, 4))
    print("F1 Score :", round(test_f1, 4))
    print("\nConfusion Matrix")
    print(confusion_matrix(truth, preds))
    print("\nClassification Report")
    print(classification_report(truth, preds, zero_division=0))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print("\nModel saved:")
    print(checkpoint_path)


if __name__ == "__main__":
    main()
