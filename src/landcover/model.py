"""The ResNet-50 classifier.

The network is loaded with ImageNet weights, its final fully connected layer is
replaced with one that outputs the ten EuroSAT classes, and the rest of the
network can be frozen or unfrozen.

``set_backbone_trainable`` is the switch between the two training stages: with
the backbone frozen only the head is updated, which is what makes the first
stage fast, and unfreezing lets the earlier layers adapt to satellite imagery
instead of natural photographs.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from . import config


def build_model(
    n_classes: int | None = None, pretrained: bool = True, seed: int = 0
) -> nn.Module:
    """Build ResNet-50 with a fresh classification head."""
    n_classes = config.N_CLASSES if n_classes is None else n_classes

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)

    # Fixed seed so the head starts from the same place on every run; without it
    # two training runs are not comparable.
    seed_everything(seed)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)
    return model


def seed_everything(seed: int = 0) -> None:
    """Fix the seeds that affect weight initialisation and batch order."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> int:
    """Freeze or unfreeze everything except the classification head.

    Returns the number of trainable parameter tensors, so a caller can confirm
    the stage was set as intended rather than assuming it.
    """
    trainable_tensors = 0
    for name, parameter in model.named_parameters():
        is_head = name.startswith("fc.")
        parameter.requires_grad = trainable if not is_head else True
        if parameter.requires_grad:
            trainable_tensors += 1
    return trainable_tensors


def group_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Total and trainable parameter counts, for the README and the console."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def save_checkpoint(
    path, model: nn.Module, extra: dict | None = None
) -> dict:
    """Write the weights and the metadata needed to rebuild the model.

    Without the architecture fields a checkpoint could not be paired back to the
    network that produced it, so the encoder settings are stored alongside the
    weights.
    """
    payload = {
        "state_dict": model.state_dict(),
        "architecture": {
            "name": "resnet50",
            "n_classes": model.fc.out_features,
            "in_features": model.fc.in_features,
            "pretrained_source": "IMAGENET1K_V2",
        },
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_checkpoint(path, device="cpu") -> tuple[nn.Module, dict]:
    """Rebuild the model described by a checkpoint and load its weights.

    A checkpoint that predates the architecture block still loads, because the
    class count falls back to the configured one.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    architecture = payload.get("architecture", {})
    n_classes = architecture.get("n_classes", config.N_CLASSES)
    model = build_model(n_classes=n_classes, pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model, payload
