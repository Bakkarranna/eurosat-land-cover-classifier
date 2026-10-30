"""Paths, dataset constants and hyperparameters for the land cover pipeline.

Every number that shapes an experiment lives here, so a run can be reproduced
from the file rather than from a shell history. Values may be overridden with
environment variables, which is what lets the test suite point the pipeline at a
tiny synthetic dataset without touching the real one.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    """Allow the data and results roots to be relocated (tests, CI smoke runs)."""
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


DATA_DIR = _env_path("LANDCOVER_DATA_DIR", PROJECT_ROOT / "data")
RESULTS_DIR = _env_path("LANDCOVER_RESULTS_DIR", PROJECT_ROOT / "results")
DOCS_DIR = _env_path("LANDCOVER_DOCS_DIR", PROJECT_ROOT / "docs")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = RAW_DIR / "EuroSAT_RGB"

# Every file the pipeline writes, defined once so the README, the console and
# the tests all agree on where an artifact lands.
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "resnet50_landcover.pt"
SMOKE_MODEL_PATH = MODELS_DIR / "smoke_test.pt"
SPLIT_PATH = RESULTS_DIR / "split.json"
TRAIN_HISTORY_PATH = RESULTS_DIR / "train_history.json"
SUMMARY_CSV_PATH = RESULTS_DIR / "dataset_summary.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "dataset_summary.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.npz"
SMOKE_MARKER_PATH = RESULTS_DIR / "smoke.txt"
SMOKE_LOG_PATH = RESULTS_DIR / "smoke.log"
FIGURES_DIR = DOCS_DIR / "figures"

# The RGB archive published with the dataset: about 92 MB, 27,000 JPEGs.
ARCHIVE_NAME = "EuroSAT_RGB.zip"
ARCHIVE_URL = "https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip"
# Recorded, not checked: the value for the release that produced the results
# in this repository is written to results/dataset_summary.json by
# landcover.data dataset_summary, and it is what a future run should match.
ARCHIVE_SHA256 = None

# The ten EuroSAT classes, in the folder order shipped in the archive. The model
# output indices are defined by this order, so it must not be reordered without
# retraining.
CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
N_CLASSES = len(CLASSES)

# Plain language descriptions, used by the README, the console and predict.py.
CLASS_DESCRIPTIONS = {
    "AnnualCrop": "fields replanted each season",
    "Forest": "dense tree cover",
    "HerbaceousVegetation": "grasslands, meadows and other non-crop ground cover",
    "Highway": "major roads and highways",
    "Industrial": "factories, warehouses and commercial zones",
    "Pasture": "land used for grazing livestock",
    "PermanentCrop": "orchards and vineyards, planted for several years",
    "Residential": "neighbourhoods and suburbs",
    "River": "rivers and streams",
    "SeaLake": "large bodies of water",
}

# Original patch size in the dataset (10 m per pixel, so each patch covers
# 640 m x 640 m on the ground).
PATCH_SIZE = 64

# ResNet-50 expects 224x224 inputs; the 64x64 patches are resized up. This adds
# no detail, it only matches the architecture the pretrained weights expect.
INPUT_SIZE = 224

# ImageNet statistics. The pretrained weights were trained on inputs normalised
# with these exact values, so anything else degrades the transfer.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Split fractions for the 27,000 patches: 80 percent train, 10 percent val,
# 10 percent test. The split is drawn per class so the class balance of the
# full dataset is preserved in all three parts.
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

SEED = 42
BATCH_SIZE = 32

# Stage one trains only the new head. Stage two unfreezes the whole network at
# ten times the lower learning rate.
HEAD_LR = 1e-3
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
LABEL_SMOOTHING = 0.0

HEAD_EPOCHS = 6
FINETUNE_EPOCHS = 4

# How many of the 27,000 training patches to use. None means all of them. The
# guide's hyperparameters assume the full set; a subset exists only so a run can
# be finished on this machine in a known amount of time.
MAX_TRAIN_IMAGES = None

# Worker threads for the data loaders. The laptop that produced the recorded
# results has four cores.
NUM_WORKERS = 4

# Intra-op torch threads for the step computation.
TORCH_THREADS = 4

# Learning rate schedule. A step decay after a few epochs lets the head settle
# before the whole network is unfrozen at a lower rate.
LR_STEP_SIZE = 3
LR_GAMMA = 0.1
