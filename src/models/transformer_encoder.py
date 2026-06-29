import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        input_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):

        x = self.transformer(x)

        return x