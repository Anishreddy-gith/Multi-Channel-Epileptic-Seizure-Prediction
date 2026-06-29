import torch
import torch.nn as nn

from src.models.eegnet import EEGNetFeatureExtractor


class AttentionPooling(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False)
        )

    def forward(self, x):
        # x: (B, T, D)
        attn_logits = self.score(x).squeeze(-1)          # (B, T)
        attn_weights = torch.softmax(attn_logits, dim=1) # (B, T)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (B, D)
        return pooled


class HybridEEGNetAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.eegnet = EEGNetFeatureExtractor()
        self.attn_pool = AttentionPooling(dim=32)

        self.classifier = nn.Sequential(
            nn.LayerNorm(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # EEGNet output: (B, 32, 1, 256)
        x = self.eegnet(x)
        x = x.squeeze(2)          # (B, 32, 256)
        x = x.permute(0, 2, 1)    # (B, 256, 32)

        x = self.attn_pool(x)     # (B, 32)
        x = self.classifier(x)    # (B, 2)
        return x