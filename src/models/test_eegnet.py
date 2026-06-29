import torch

from eegnet import EEGNetFeatureExtractor

model = EEGNetFeatureExtractor()

x = torch.randn(
    8,
    1,
    18,
    1024
)

y = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)