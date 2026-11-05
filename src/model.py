"""Model: ResNet18 with transfer learning."""

import torch.nn as nn
from torchvision import models


def build_model(pretrained=True):
    """ResNet18 pretrained on ImageNet with a new 2-class output layer.

    Transfer learning: the convolutional layers already know how to detect edges,
    textures and shapes (learnt from 1.2 million ImageNet photos). They are frozen,
    and only the last layer is trained to decide Normal vs Pneumonia. This needs
    far fewer images and much less time than training the whole network.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze the pretrained layers: their weights will not change during training
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last layer (1000 ImageNet classes) with a new one with 2 outputs.
    # New layers are trainable by default, so only these ~1 000 weights are learnt.
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
