"""
ResNet18-based multitask model for tactile sensing.

- Backbone: ResNet18 (pretrained on ImageNet or not, configurable)
- Heads: force regression (Fx, Fy, Fz, Mx, My, Mz, Ft), shape classification, contact point regression
- Usage: Set num_shape_classes for your dataset. Select pretrained weights as needed.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18

class IsoNet(nn.Module):
    def __init__(self, num_shape_classes, pretrained=True):
        super(IsoNet, self).__init__()
        if pretrained:
            base = resnet18(weights="IMAGENET1K_V1")
        else:
            base = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # Output: (batch, 512, 1, 1)
        self.flatten = nn.Flatten()  # To shape (batch, 512)
        self.force_head = nn.Linear(512, 7)         # Fx, Fy, Fz, Mx, My, Mz, Ft
        self.class_head = nn.Linear(512, num_shape_classes)
        self.point_head = nn.Linear(512, 2)         # x, y contact point

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        forces = self.force_head(x)
        shape_logits = self.class_head(x)
        contact_point = self.point_head(x)
        return forces, shape_logits, contact_point
