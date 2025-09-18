import torch
import torch.nn as nn

# Custom convolutional architecture for multi-task tactile sensing.
# Includes feature extraction blocks and three output heads.
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool=True):
        """
        Convolutional block: Conv2d + ReLU (+ optional MaxPool2d).
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Convolution kernel size.
            pool (bool): Whether to apply MaxPool2d.
        """
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the block.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after conv, activation, and pooling.
        """
        return self.block(x)

class IsoNet(nn.Module):
    def __init__(self, num_shape_classes):
        """
        Initialize IsoNet with custom CNN backbone and three output heads.
        Args:
            num_shape_classes (int): Number of shape classes for classification.
        """
        super(IsoNet, self).__init__()
        self.cnn = nn.Sequential(
            ConvBlock(3, 16),       # (B, 16, 112, 112)
            ConvBlock(16, 32),      # (B, 32, 56, 56)
            ConvBlock(32, 64),      # (B, 64, 28, 28)
            ConvBlock(64, 128),     # (B, 128, 14, 14)
            nn.AdaptiveAvgPool2d((1, 1))  # (B, 128, 1, 1)
        )
        self.flatten = nn.Flatten()  # (B, 128)
        # Multi-task heads
        self.force_head = nn.Linear(128, 7)           # Force regression (Fx, Fy, Fz, Mx, My, Mz, Ft)
        self.class_head = nn.Linear(128, num_shape_classes)  # Shape classification
        self.point_head = nn.Linear(128, 2)           # Contact point regression (X, Y)

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input image tensor (B, 3, 224, 224)
        Returns:
            tuple: (forces, shape_logits, contact_point)
        """
        x = self.cnn(x)
        x = self.flatten(x)
        forces = self.force_head(x)
        shape_logits = self.class_head(x)
        contact_point = self.point_head(x)
        return forces, shape_logits, contact_point
