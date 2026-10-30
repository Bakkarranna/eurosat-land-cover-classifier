"""Score a trained checkpoint on the held-out test split.

The test patches were not used for training and were not used for model
selection, so the number this writes is the honest one. It is the accuracy a
reader should quote: not the best validation accuracy seen during training.

Outputs ``results/metrics.json`` with overall accuracy, top-5 accuracy, the
per-class precision / recall / F1 table, the confusion matrix, and the worst
confusions, plus ``results/predictions.npz`` holding the scores and labels for
every test patch so figures can be rebuilt without loading the model again.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config, data
from . import metrics as metrics_module
from . import model as model_module
from .dataset import LandcoverDataset, build_eval_transform
from .train import resolve_device


def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader and return the scores and the true labels."""
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[int] = []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            scores.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.extend(batch_labels.tolist())
    return np.concatenate(scores, axis=0), np.asarray(labels, dtype=np.int64)


def evaluate_checkpoint(
    checkpoint: Path | None = None, batch_size: int = 64, workers: int = 0
) -> dict:
    """Load a checkpoint, score the test split and write the metrics files."""
    checkpoint = checkpoint or config.MODEL_PATH
    device = resolve_device()
    model, payload = model_module.load_checkpoint(checkpoint, device=str(device))

    split = data.split_dataset(data.index_dataset())
    test_samples = split["test"]
    loader = DataLoader(
        LandcoverDataset(test_samples, transform=build_eval_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    scores, labels = collect_predictions(model, loader, device)
    predicted = scores.argmax(axis=1)

    matrix = metrics_module.confusion_matrix(labels, predicted, config.N_CLASSES)
    per_class = metrics_module.per_class_prf(matrix)
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "checkpoint": str(checkpoint),
        "n_test": int(len(labels)),
        "classes": config.CLASSES,
        "accuracy": per_class["accuracy"],
        "macro_f1": per_class["macro_f1"],
        "top5_accuracy": metrics_module.top_k_accuracy(scores, labels, 5),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "classification_report": metrics_module.format_classification_report(
            matrix, config.CLASSES, per_class
        ),
        "worst_confusions": metrics_module.worst_pairs(matrix, config.CLASSES),
        "run": payload.get("run", {}),
    }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    np.savez_compressed(
        config.PREDICTIONS_PATH,
        scores=scores.astype(np.float32),
        labels=labels,
        predicted=predicted,
    )
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the land cover classifier")
    parser.add_argument("--checkpoint", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    report = evaluate_checkpoint(args.checkpoint, batch_size=args.batch_size)
    print(report["classification_report"])
    print(
        f"\ntest accuracy {report['accuracy']:.4f}  "
        f"top-5 {report['top5_accuracy']:.4f}  n={report['n_test']}"
    )


if __name__ == "__main__":
    main()
