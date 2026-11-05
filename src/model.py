"""ResNet18 with transfer learning: frozen ImageNet backbone + new 2-class layer."""

import torch.nn as nn
from torchvision import models


def build_model(pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    for param in model.parameters():  # keep what ResNet learnt on ImageNet
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)  # only this layer is trained
    return model
