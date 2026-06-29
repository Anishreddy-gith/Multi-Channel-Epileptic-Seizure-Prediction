import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.project_paths import ensure_project_storage, output_path, print_storage_summary

ensure_project_storage()

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

cm = np.array([
    [296, 104],
    [108, 42]
])

labels = ["Interictal", "Preictal"]

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, cmap="Blues", values_format="d")

plt.title("Hybrid EEGNet + Transformer Confusion Matrix")
plt.tight_layout()
fig_path = output_path("figures/hybrid_confusion_matrix.png")
fig_path.parent.mkdir(parents=True, exist_ok=True)
print_storage_summary({"figure": fig_path})
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {fig_path}")
