"""Inference for one or more image files.

Point it at a JPEG and it prints the ranked classes with their probabilities.
It loads the same saved checkpoint the evaluation uses, so a prediction comes
from the model that produced the reported metrics, not a freshly initialised
network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import config
from . import model as model_module
from .dataset import build_eval_transform
from .train import resolve_device


def load_images(paths: list[Path]) -> torch.Tensor:
    """Decode the given image files into one float tensor."""
    transform = build_eval_transform()
    tensors = [transform(Image.open(path).convert("RGB")) for path in paths]
    return torch.stack(tensors, dim=0)


def predict_paths(paths: list[Path], checkpoint: Path | None = None, top_k: int = 5):
    """Return the top-k class names and probabilities for each image path."""
    checkpoint = checkpoint or config.MODEL_PATH
    device = resolve_device()
    model, _ = model_module.load_checkpoint(checkpoint, device=str(device))
    model.eval()

    batch = load_images(paths).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=1).cpu().numpy()

    results = []
    for path, row in zip(paths, probabilities, strict=True):
        order = np.argsort(row)[::-1][:top_k]
        results.append(
            {
                "image": str(path),
                "predictions": [
                    {"label": config.CLASSES[index], "probability": float(row[index])}
                    for index in order
                ],
                "label": config.CLASSES[int(row.argmax())],
            }
        )
    return results


def _expand(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        candidate = Path(item)
        paths.extend(sorted(candidate.glob("*.jpg")) if candidate.is_dir() else [candidate])
    return paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Classify EuroSAT image patches")
    parser.add_argument("images", nargs="+", help="JPEG files or directories of them")
    parser.add_argument("--checkpoint", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)

    paths = _expand(args.images)
    if not paths:
        parser.error("no images found")
    results = predict_paths(paths, args.checkpoint, top_k=args.top_k)

    if args.json:
        print(json.dumps(results, indent=2))
        return
    for result in results:
        name = Path(result["image"]).name
        top = ", ".join(
            f"{item['label']} {item['probability']:.3f}"
            for item in result["predictions"][:3]
        )
        print(f"{name}: {top}")


if __name__ == "__main__":
    main()
