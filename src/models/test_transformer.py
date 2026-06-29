import torch

from transformer_encoder import TransformerEncoder

model = TransformerEncoder()

x = torch.randn(
    8,
    256,
    32
)

y = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)