"""The two stage training loop.

Stage one trains only the new classification head while the rest of ResNet-50
stays frozen, which is fast because no gradients flow through 23 million
backbone parameters. Stage two unfreezes the whole network at a tenth of the
learning rate so the backbone can adapt from natural photographs to satellite
patches without losing what it already knows.

Each stage is a configurable number of epochs, and the command line flags exist
so a run can be sized to the machine it is on. Everything the run touched is
recorded: the exact split, the hyperparameters, the wall clock time and the
per-epoch metrics.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config, data
from . import metrics as metrics_module
from . import model as model_module
from .dataset import LandcoverDataset, build_eval_transform, build_train_transform


def resolve_device() -> torch.device:
    """Prefer CUDA, then Apple silicon, then CPU, and say which one is used."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(samples, transform, shuffle: bool, batch_size: int, workers: int):
    return DataLoader(
        LandcoverDataset(samples, transform=transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
    )


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> dict:
    """One pass over a loader. In eval mode no gradients are taken or stored."""
    model.train(train)
    running_loss = 0.0
    seen = 0
    correct = 0
    all_labels: list[int] = []
    all_predicted: list[int] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            predicted = outputs.argmax(dim=1)
            seen += labels.size(0)
            correct += int((predicted == labels).sum().item())
            all_labels.extend(labels.detach().cpu().tolist())
            all_predicted.extend(predicted.detach().cpu().tolist())

    matrix = metrics_module.confusion_matrix(all_labels, all_predicted, config.N_CLASSES)
    per_class = metrics_module.per_class_prf(matrix)
    return {
        "loss": running_loss / seen,
        "accuracy": correct / seen,
        "macro_f1": per_class["macro_f1"],
        "confusion_matrix": matrix.tolist(),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EuroSAT land cover classifier")
    parser.add_argument("--head-epochs", type=int, default=config.HEAD_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=config.FINETUNE_EPOCHS)
    parser.add_argument("--epochs", type=int, default=None,
                        help="shortcut for --head-epochs when stage 2 is skipped")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--max-train-images", type=int, default=config.MAX_TRAIN_IMAGES)
    parser.add_argument("--skip-finetune", action="store_true",
                        help="run stage one only")
    parser.add_argument("--out", type=Path, default=config.MODEL_PATH,
                        help="checkpoint destination")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--limit-val", type=int, default=None,
                        help="cap the validation set, for fast smoke runs")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    head_epochs = args.epochs if args.epochs is not None else args.head_epochs

    device = resolve_device()
    # Intra-op threads. The data loader workers are separate: pick whichever
    # combination fills the cores, and record what was used.
    torch.set_num_threads(config.TORCH_THREADS)
    print(f"device: {device}")

    model_module.seed_everything(args.seed)
    data.prepare_dataset()
    index = data.index_dataset()
    split = data.split_dataset(index, seed=args.seed)

    train_samples = data.optional_subset(split["train"], args.max_train_images)
    val_samples = split["val"]
    if args.limit_val is not None:
        val_samples = val_samples[: args.limit_val]

    data.write_split_json(split, config.SPLIT_PATH)
    print(
        f"train {len(train_samples)}  val {len(val_samples)}  "
        f"test {len(split['test'])}"
    )

    train_loader = make_loader(
        train_samples, build_train_transform(), True, args.batch_size, args.workers
    )
    val_loader = make_loader(
        val_samples, build_eval_transform(), False, args.batch_size, args.workers
    )

    model = model_module.build_model(seed=args.seed).to(device)
    criterion = nn.CrossEntropyLoss()

    # Stage one: head only. The learning rate is the standard starting point for
    # a fresh linear head on frozen features.
    trainable = model_module.set_backbone_trainable(model, False)
    print(f"stage 1: {trainable} trainable tensors (head only)")
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.HEAD_LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )

    history: list[dict] = []
    started = time.time()
    for stage, epochs, lr in (
        ("head", head_epochs, config.HEAD_LR),
        ("finetune", args.finetune_epochs if not args.skip_finetune else 0, config.FINETUNE_LR),
    ):
        if epochs <= 0:
            continue
        if stage == "finetune":
            trainable = model_module.set_backbone_trainable(model, True)
            print(f"stage 2: {trainable} trainable tensors (whole network)")
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=config.MOMENTUM,
                weight_decay=config.WEIGHT_DECAY,
            )

        for epoch in range(epochs):
            train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True)
            val_metrics = run_epoch(model, val_loader, criterion, None, device, False)
            record = {
                "stage": stage,
                "epoch": len(history) + 1,
                "train_loss": round(train_metrics["loss"], 4),
                "train_accuracy": round(train_metrics["accuracy"], 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_accuracy": round(val_metrics["accuracy"], 4),
                "val_macro_f1": round(val_metrics["macro_f1"], 4),
            }
            history.append(record)
            print(
                f"[{stage} {epoch + 1}/{epochs}] loss {record['train_loss']:.4f} "
                f"acc {record['train_accuracy']:.4f}  val acc {record['val_accuracy']:.4f}"
            )

    elapsed = time.time() - started

    counts = model_module.group_parameter_counts(model)
    record = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "seconds": round(elapsed, 1),
        "head_epochs": head_epochs,
        "finetune_epochs": 0 if args.skip_finetune else args.finetune_epochs,
        "batch_size": args.batch_size,
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "seed": args.seed,
        "threads": config.TORCH_THREADS,
        "workers": args.workers,
        "history": history,
        "parameters": counts,
    }
    model_module.save_checkpoint(args.out, model, {"run": record})
    with open(config.TRAIN_HISTORY_PATH, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(f"saved {args.out} in {elapsed / 60:.1f} min")
    return record


if __name__ == "__main__":
    main()
