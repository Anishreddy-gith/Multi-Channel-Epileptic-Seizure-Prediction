import torch
import torch.nn as nn


class EEGNetFeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        self.temporal_conv = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=(1, 64),
                padding=(0, 32),
                bias=False
            ),

            nn.BatchNorm2d(16)
        )

        self.spatial_conv = nn.Sequential(

            nn.Conv2d(
                16,
                32,
                kernel_size=(18, 1),
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ELU(),

            nn.AvgPool2d((1, 4)),

            nn.Dropout(0.25)
        )

    def forward(self, x):

        x = self.temporal_conv(x)

        x = self.spatial_conv(x)

        return x