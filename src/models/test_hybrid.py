import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
from src.models.hybrid_eegnet_transformer import HybridEEGNetTransformer

model = HybridEEGNetTransformer()

x = torch.randn(8, 1, 18, 1024)
y = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)

assert y.shape == (8, 2)

print("\nHybrid EEGNet + Transformer Test Passed")