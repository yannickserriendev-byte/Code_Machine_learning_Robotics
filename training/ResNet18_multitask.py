import torch
import torch.nn as nn
from torchvision.models import resnet18

class IsoNet(nn.Module):
    def __init__(self, num_shape_classes):
        super(IsoNet, self).__init__()

        # Load ResNet18 WITHOUT pretrained weights
        base = resnet18(weights=None)  # or pretrained=False if using older Torch
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        self.flatten = nn.Flatten()
        self.force_head = nn.Linear(512, 7)
        self.class_head = nn.Linear(512, num_shape_classes)
        self.point_head = nn.Linear(512, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        return self.force_head(x), self.class_head(x), self.point_head(x)

