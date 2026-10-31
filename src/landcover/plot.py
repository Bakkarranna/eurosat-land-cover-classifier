"""Save the static figures the README refers to.

Everything here reads from ``results/predictions.npz`` and ``results/metrics.json``
rather than re-running the model, so the figures and the numbers can never
disagree. Run it after evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from . import config
from .data import index_dataset, split_dataset
from .dataset import build_eval_transform
from .model import load_checkpoint
from .train import resolve_device


def plot_confusion_matrix(metrics_path: Path, output_path: Path) -> None:
    """Draw the confusion matrix with counts and per-row normalisation."""
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    classes = metrics["classes"]

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(
        matrix, row_totals, out=np.zeros_like(matrix, dtype=np.float64), where=row_totals > 0
    )

    height = 0.45 * len(classes) + 3.0
    figure, axis = plt.subplots(figsize=(height * 0.95, height))
    image = axis.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="share of true class")

    threshold = 0.5 * np.max(normalised) if np.max(normalised) > 0 else 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            value = int(matrix[i, j])
            share = f"{normalised[i, j]:.2f}"
            axis.text(
                j,
                i,
                f"{value}\n{share}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if normalised[i, j] > threshold else "black",
            )
    axis.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    axis.set_title(f"Confusion matrix, test accuracy {metrics['accuracy']:.3f}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_sample_images(
    output_path: Path, per_class: int = 5, checkpoint: Path | None = None
) -> None:
    """Show a few real test patches per class so the reader sees the input."""
    split = split_dataset(index_dataset())
    device = resolve_device()
    model, _ = load_checkpoint(checkpoint or config.MODEL_PATH, device=str(device))

    chosen: list[tuple[np.ndarray, str, str]] = []
    rows = 1
    for class_index, class_name in enumerate(config.CLASSES):
        samples = [p for p, i in split["test"] if i == class_index][:per_class]
        rows = max(rows, len(samples))
        for path in samples:
            tensor = build_eval_transform()(Image.open(path).convert("RGB"))
            array = (tensor * 0.25 + 0.45).clamp(0, 1).permute(1, 2, 0).numpy()
            with torch.no_grad():
                prediction = model(tensor.unsqueeze(0).to(device)).argmax(dim=1).item()
            chosen.append((array, class_name, config.CLASSES[prediction]))

    columns = len(config.CLASSES)
    # The grid follows the patches that were actually selected: a test split with
    # fewer than ``per_class`` patches in a class (a smoke run has two) would
    # otherwise leave the requested rows empty and mismatch the axes.
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.25, rows * 1.35))
    flat_axes = np.ravel(axes)
    for axis, (array, true, predicted) in zip(flat_axes, chosen, strict=False):
        axis.imshow(array)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel(
            f"{predicted[:12]} {'ok' if true == predicted else 'X'}",
            fontsize=6,
            color="#333" if true == predicted else "#b00",
        )
    for axis in flat_axes[len(chosen) :]:
        axis.axis("off")
    for axis, class_name in zip(flat_axes[:columns], config.CLASSES, strict=True):
        axis.set_title(class_name[:12], fontsize=6)
    figure.suptitle("Test patches with the predicted class below each one", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_class_distribution(output_path: Path) -> None:
    """Bar chart of patches per class, so the reader sees the balance."""
    summary = json.loads((config.SUMMARY_JSON_PATH).read_text(encoding="utf-8"))
    classes = list(summary["classes"])
    counts = [summary["classes"][name] for name in classes]
    figure, axis = plt.subplots(figsize=(9, 3.6))
    bars = axis.bar(range(len(classes)), counts, color="#3b7d5a")
    axis.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    axis.set_ylabel("patches")
    axis.set_title(f"EuroSAT RGB: {sum(counts)} patches in {len(classes)} classes")
    for bar, count in zip(bars, counts, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, count, str(count), ha="center", va="bottom", fontsize=7)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_history(output_path: Path) -> None:
    """Plot the training and validation loss and accuracy curves."""
    metrics = json.loads((config.METRICS_PATH).read_text(encoding="utf-8"))
    history = metrics.get("run", {}).get("history", [])
    if not history:
        print("no training history in metrics.json; skipping the curves")
        return
    epochs = [entry["epoch"] for entry in history]
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(9, 3.6))
    loss_axis.plot(epochs, [e["train_loss"] for e in history], label="train")
    loss_axis.plot(epochs, [e["val_loss"] for e in history], label="val")
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("loss")
    loss_axis.legend(frameon=False)
    accuracy_axis.plot(epochs, [e["train_accuracy"] for e in history], label="train")
    accuracy_axis.plot(epochs, [e["val_accuracy"] for e in history], label="val")
    accuracy_axis.set_xlabel("epoch")
    accuracy_axis.set_ylabel("accuracy")
    accuracy_axis.set_ylim(0, 1)
    accuracy_axis.legend(frameon=False)
    figure.suptitle("Training curves", fontsize=10)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=config.METRICS_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="checkpoint for the sample patch figure; defaults to config.MODEL_PATH")
    args = parser.parse_args(argv)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(args.metrics, config.FIGURES_DIR / "confusion_matrix.png")
    plot_history(config.FIGURES_DIR / "history.png")
    plot_class_distribution(config.FIGURES_DIR / "class_distribution.png")
    plot_sample_images(config.FIGURES_DIR / "sample_patches.png", checkpoint=args.checkpoint)
    for name in ("confusion_matrix.png", "history.png", "class_distribution.png", "sample_patches.png"):
        print(f"wrote {config.FIGURES_DIR / name}")


if __name__ == "__main__":
    main()
