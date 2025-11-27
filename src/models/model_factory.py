"""Model factory for creating different architectures"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

def create_model(
    architecture: str = "resnet50",
    num_classes: int = 6,
    pretrained: bool = True,
    weights: Optional[str] = "IMAGENET1K_V2"
) -> nn.Module:
    """Create model with specified architecture"""
    
    architecture = architecture.lower()
    
    # ResNet models
    if architecture == "resnet18":
        model = models.resnet18(weights=weights if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif architecture == "resnet50":
        model = models.resnet50(weights=weights if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif architecture == "resnet101":
        model = models.resnet101(weights=weights if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # EfficientNet models
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif architecture == "efficientnet_b3":
        model = models.efficientnet_b3(weights=weights if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # MobileNet
    elif architecture == "mobilenet_v3":
        model = models.mobilenet_v3_large(weights=weights if pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model

