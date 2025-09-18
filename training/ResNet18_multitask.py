import torch
import torch.nn as nn
from torchvision.models import resnet18

# =============================================================
# ResNet18_multitask.py
# -------------------------------------------------------------
# Defines IsoNet: Multi-task model for tactile sensing using
# ResNet18 backbone. Outputs force, shape class, and contact point.
# =============================================================

class IsoNet(nn.Module):
    """
    Multi-task neural network for tactile sensing using ResNet18 backbone.
    Outputs force vector, shape class logits, and contact point coordinates.
    """
    def __init__(self, num_shape_classes):
        """
        Initialize IsoNet with ResNet18 backbone and three output heads.
        Args:
            num_shape_classes (int): Number of shape classes for classification.
        """
        super(IsoNet, self).__init__()
        # Load ResNet18 WITHOUT pretrained weights
        base = resnet18(weights=None)  # or pretrained=False if using older Torch
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.flatten = nn.Flatten()
        self.force_head = nn.Linear(512, 7)  # Force regression (Fx, Fy, Fz, Mx, My, Mz, Ft)
        self.class_head = nn.Linear(512, num_shape_classes)  # Shape classification
        self.point_head = nn.Linear(512, 2)  # Contact point regression (X, Y)

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input image tensor (B, 3, 224, 224)
        Returns:
            tuple: (forces, shape_logits, contact_point)
        """
        x = self.backbone(x)
        x = self.flatten(x)
        return self.force_head(x), self.class_head(x), self.point_head(x)

