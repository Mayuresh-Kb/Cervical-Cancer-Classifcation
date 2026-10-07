"""Compatibility helper for VGG16 from torchvision.

New experiments should select ``--model vgg16`` in train.py. This module is
kept only for code that previously imported ``VGG16`` from this location.
"""
from torch import nn
from torchvision import models


def VGG16(num_classes: int = 5, pretrained: bool = False) -> nn.Module:
    """Return torchvision's maintained VGG16 with a task-specific head."""
    model = models.vgg16(weights="DEFAULT" if pretrained else None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model
