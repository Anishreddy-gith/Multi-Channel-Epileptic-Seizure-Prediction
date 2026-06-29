import torch
import torch.nn as nn

from src.models.eegnet import EEGNetFeatureExtractor
from src.models.transformer_encoder import TransformerEncoder


class HybridEEGNetTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.eegnet = EEGNetFeatureExtractor()

        self.seq_len = 256
        self.embed_dim = 32

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.seq_len + 1, self.embed_dim)
        )
        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(
            input_dim=32,
            num_heads=4,
            num_layers=2,
            dropout=0.1
        )

        self.norm = nn.LayerNorm(self.embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.eegnet(x)          # (B, 32, 1, 256)
        x = x.squeeze(2)            # (B, 32, 256)
        x = x.permute(0, 2, 1)      # (B, 256, 32)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)   # (B, 1, 32)
        x = torch.cat((cls_tokens, x), dim=1)                   # (B, 257, 32)

        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)     # (B, 257, 32)

        x = x[:, 0]                 # CLS token
        x = self.norm(x)

        x = self.classifier(x)      # (B, 2)
        return x