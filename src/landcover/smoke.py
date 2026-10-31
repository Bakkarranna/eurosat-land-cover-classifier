"""A tiny end to end run so the pipeline can be validated without the dataset.

It writes synthetic patches (random noise tinted by class), trains for two
epochs so the accuracy it prints is meaningless, and runs the same evaluation,
figure and prediction paths the real run uses. The point is to prove every
module loads, every stride connects, and every output file lands where the
README says it does, in about a minute on a CPU.

The synthetic dataset lives in a temporary directory that the pipeline is
pointed at through the landcover.config module, so a smoke run never touches
the real EuroSAT images on disk and never needs the 90 MB download.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from . import data as data_module
from . import evaluate as evaluate_module
from . import model as model_module
from . import plot as plot_module
from . import predict as predict_module
from . import train as train_module


def build_synthetic_dataset(root: Path, per_class: int = 20, size: int = 64) -> Path:
    """Write ``per_class`` noisy patches per class, each tinted by its class.

    The tint is what lets a model learn anything at all from the noise: without
    it, class and noise are independent and the accuracy stays at chance.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEED)
    for class_index, class_name in enumerate(config.CLASSES):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        # A distinct hue per class, on a 0.75..1.25 scale around mid grey.
        tint = np.array(
            [
                0.75 + 0.5 * ((class_index % 3) / 2.0),
                0.75 + 0.5 * (((class_index + 1) % 3) / 2.0),
                0.75 + 0.5 * (((class_index + 2) % 3) / 2.0),
            ]
        )
        for j in range(per_class):
            noise = rng.random((size, size, 3), dtype=np.float32) * 60 + 40
            patch = np.clip(noise * tint, 0, 255).astype(np.uint8)
            Image.fromarray(patch, "RGB").save(class_dir / f"synthetic_{j}.jpg", quality=92)
    return root


def _point_at_temporary_root(work: Path) -> None:
    """Redirect every pipeline path into a temporary directory.

    landcover.config resolves its paths once at import time, so the modules
    cannot be told about another directory through an environment variable
    after the fact. Overriding the attributes on the imported config module is
    enough: every module reads config.SOMETHING when it runs, not at import.
    """
    config.DATA_DIR = work / "data"
    config.RESULTS_DIR = work / "results"
    config.DOCS_DIR = work / "docs"
    config.RAW_DIR = config.DATA_DIR / "raw"
    config.IMAGES_DIR = config.RAW_DIR / "EuroSAT_RGB"
    config.MODELS_DIR = work / "models"
    config.MODEL_PATH = config.MODELS_DIR / "resnet50_landcover.pt"
    config.SMOKE_MODEL_PATH = config.MODELS_DIR / "smoke_test.pt"
    config.SPLIT_PATH = config.RESULTS_DIR / "split.json"
    config.TRAIN_HISTORY_PATH = config.RESULTS_DIR / "train_history.json"
    config.SUMMARY_CSV_PATH = config.RESULTS_DIR / "dataset_summary.csv"
    config.SUMMARY_JSON_PATH = config.RESULTS_DIR / "dataset_summary.json"
    config.METRICS_PATH = config.RESULTS_DIR / "metrics.json"
    config.PREDICTIONS_PATH = config.RESULTS_DIR / "predictions.npz"
    config.SMOKE_MARKER_PATH = config.RESULTS_DIR / "smoke.txt"
    config.SMOKE_LOG_PATH = config.RESULTS_DIR / "smoke.log"
    config.FIGURES_DIR = config.DOCS_DIR / "figures"
    for directory in (config.RESULTS_DIR, config.MODELS_DIR, config.FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def main(argv=None) -> dict:
    """Build synthetic patches, train, evaluate, plot and predict with them."""
    parser = argparse.ArgumentParser(description="Run the pipeline on synthetic patches")
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        _point_at_temporary_root(work)
        root = build_synthetic_dataset(config.IMAGES_DIR, per_class=args.per_class)
        print(f"wrote {args.per_class * config.N_CLASSES} synthetic patches to {root}")

        # The class distribution figure is drawn from the dataset summary, so the
        # summary has to exist before the figures are rebuilt.
        summary = data_module.dataset_summary()
        data_module.write_summary_csv(summary, config.SUMMARY_CSV_PATH)
        data_module.write_summary_json(summary, config.SUMMARY_JSON_PATH)

        record = train_module.main(
            [
                "--epochs",
                str(args.epochs),
                "--skip-finetune",
                "--batch-size",
                str(args.batch_size),
                "--workers",
                "0",
                "--out",
                str(config.SMOKE_MODEL_PATH),
            ]
        )
        report = evaluate_module.evaluate_checkpoint(checkpoint=config.SMOKE_MODEL_PATH, workers=0)
        # An explicit argv keeps plot's parser off smoke's own flags, and points
        # the sample patch figure at the checkpoint this run just trained rather
        # than the full run's, which does not exist yet.
        plot_module.main(["--checkpoint", str(config.SMOKE_MODEL_PATH)])
        sample = sorted(root.glob("*/*.jpg"))[0]
        predictions = predict_module.predict_paths([sample], config.SMOKE_MODEL_PATH)
        model, payload = model_module.load_checkpoint(config.SMOKE_MODEL_PATH)
        model.eval()
        config.SMOKE_MARKER_PATH.write_text(
            f"smoke ok: {report['n_test']} test patches, accuracy {report['accuracy']:.4f}\n",
            encoding="utf-8",
        )
        print(f"smoke ok: accuracy {report['accuracy']:.4f} on synthetic patches")
        print(f"wrote {config.SMOKE_MARKER_PATH}")
        return {"train": record, "report": report, "predictions": predictions, "payload": payload}


if __name__ == "__main__":
    main()
