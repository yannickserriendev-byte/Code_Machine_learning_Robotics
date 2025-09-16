"""
SimpleCNN-based multitask model for tactile sensing.

- Backbone: 4-layer CNN with ReLU and MaxPool
- Heads: force regression (Fx, Fy, Fz, Mx, My, Mz, Ft), shape classification, contact point regression
- Usage: Set num_shape_classes for your dataset.
"""
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool=True):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_shape_classes):
        super(SimpleCNN, self).__init__()
        self.cnn = nn.Sequential(
            ConvBlock(3, 16),       # -> (B, 16, 112, 112)
            ConvBlock(16, 32),      # -> (B, 32, 56, 56)
            ConvBlock(32, 64),      # -> (B, 64, 28, 28)
            ConvBlock(64, 128),     # -> (B, 128, 14, 14)
            nn.AdaptiveAvgPool2d((1, 1))  # -> (B, 128, 1, 1)
        )
        self.flatten = nn.Flatten()  # -> (B, 128)
        self.force_head = nn.Linear(128, 7)
        self.class_head = nn.Linear(128, num_shape_classes)
        self.point_head = nn.Linear(128, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        forces = self.force_head(x)
        shape_logits = self.class_head(x)
        contact_point = self.point_head(x)
        return forces, shape_logits, contact_point
