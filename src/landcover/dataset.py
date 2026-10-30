"""Torch datasets and the image transforms.

The images are 64x64 JPEGs and the network wants 224x224 float tensors scaled
with ImageNet statistics. Two transform sets exist because they do different
jobs:

* the training transform adds random flips. A satellite patch has no canonical
  orientation, so a flipped copy is still a valid example of the same class and
  the extra variation costs nothing.
* the evaluation transform only resizes and normalises, so validation and test
  images are scored exactly as they were collected.

They must not be swapped: advertising the test score on augmented images would
not describe the model that ships.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from . import config

SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png")


def build_train_transform() -> transforms.Compose:
    """Resize, flip, tensorise and normalise with ImageNet statistics."""
    return transforms.Compose(
        [
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    """The same preparation without augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


class LandcoverDataset(Dataset):
    """A list of ``(path, class_index)`` pairs, decoded on demand."""

    def __init__(self, samples, transform=None) -> None:
        self.samples = [(Path(p), int(c)) for p, c in samples]
        self.transform = transform or build_eval_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, class_index = self.samples[index]
        try:
            image = Image.open(path)
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Could not read {path}: {error}") from error
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, class_index

    def labels(self) -> list[int]:
        return [c for _, c in self.samples]

    def paths(self) -> list[str]:
        return [str(p) for p, _ in self.samples]


def load_image(path: str | Path) -> torch.Tensor:
    """Prepare one image file for inference, as a batch of one."""
    image = Image.open(path).convert("RGB")
    tensor = build_eval_transform()(image)
    return tensor.unsqueeze(0)
