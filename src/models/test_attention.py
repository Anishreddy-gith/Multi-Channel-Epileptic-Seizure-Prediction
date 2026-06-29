import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
from src.models.hybrid_attention import HybridEEGNetAttention

model = HybridEEGNetAttention()

x = torch.randn(8, 1, 18, 1024)
y = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)

assert y.shape == (8, 2)

print("\nHybrid EEGNet + Attention Test Passed")