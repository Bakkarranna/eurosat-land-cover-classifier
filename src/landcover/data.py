"""Dataset discovery, archive handling and the reproducible 80/10/10 split.

The EuroSAT archive is a zip holding ten class folders of JPEGs. This module
extracts it if needed, indexes every patch as ``(path, class_index)``, and cuts
the index into train / val / test.

The split is drawn per class with a fixed seed, and the chosen file names are
written to ``results/split.json`` so a later run can verify it used the same
partition instead of trusting a seed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import zipfile
from pathlib import Path

from . import config


class DatasetError(RuntimeError):
    """Raised when the dataset is missing, incomplete or corrupt."""


def archive_path() -> Path:
    return config.RAW_DIR / config.ARCHIVE_NAME


def download_archive(destination: Path | None = None, force: bool = False) -> Path:
    """Fetch the RGB archive from Zenodo. Skips the download if it is present."""
    destination = destination or archive_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination

    import urllib.request

    print(f"Downloading {config.ARCHIVE_URL}")
    with urllib.request.urlopen(config.ARCHIVE_URL) as response, open(
        destination, "wb"
    ) as handle:
        total = int(response.headers.get("Content-Length", 0))
        copied = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            copied += len(chunk)
            if total:
                print(f"  {copied / 1e6:.0f} MB of {total / 1e6:.0f} MB", end="\r")
    print()
    return destination


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive: Path | None = None, destination: Path | None = None) -> Path:
    """Unzip the RGB archive into data/raw if it is not extracted already."""
    archive = archive or archive_path()
    destination = destination or config.IMAGES_DIR
    if destination.exists() and any(destination.glob("*/*.jpg")):
        return destination

    if not archive.exists():
        raise DatasetError(
            f"Neither {destination} nor {archive} exists. Run "
            f"`python -m landcover.data` to download and extract the dataset."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive.name} ({archive.stat().st_size / 1e6:.0f} MB)")
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination.parent)
    return destination


def index_dataset(images_dir: Path | None = None) -> list[tuple[Path, int]]:
    """Return every patch as ``(path, class_index)``, in a stable order.

    Sorting keeps the index identical across machines and filesystems, which is
    what makes the seeded split reproducible.
    """
    images_dir = images_dir or config.IMAGES_DIR
    if not images_dir.exists():
        raise DatasetError(f"Dataset directory not found: {images_dir}")

    index: list[tuple[Path, int]] = []
    for class_index, class_name in enumerate(config.CLASSES):
        class_dir = images_dir / class_name
        if not class_dir.is_dir():
            raise DatasetError(f"Missing class directory: {class_dir}")
        paths = sorted(p for p in class_dir.glob("*.jpg") if p.is_file())
        if not paths:
            raise DatasetError(f"No JPEG patches in {class_dir}")
        index.extend((p, class_index) for p in paths)
    return index


def class_counts(index: list[tuple[Path, int]]) -> dict[str, int]:
    """Count patches per class name, in class order."""
    counts = [0] * config.N_CLASSES
    for _, class_index in index:
        counts[class_index] += 1
    return {config.CLASSES[i]: counts[i] for i in range(config.N_CLASSES)}


def split_dataset(
    index: list[tuple[Path, int]], seed: int | None = None
) -> dict[str, list[tuple[Path, int]]]:
    """Split the index into train / val / test, stratified by class.

    Randomly assigning each patch to one of the three parts matters: a split
    that grouped the easy patches into train and the hard ones into test would
    flatter the final score.
    """
    seed = config.SEED if seed is None else seed
    rng = random.Random(seed)

    by_class: dict[int, list[tuple[Path, int]]] = {
        i: [] for i in range(config.N_CLASSES)
    }
    for entry in index:
        by_class[entry[1]].append(entry)

    split: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    for class_index in range(config.N_CLASSES):
        samples = by_class[class_index]
        rng.shuffle(samples)
        n = len(samples)
        n_train = int(round(config.TRAIN_FRACTION * n))
        n_val = int(round(config.VAL_FRACTION * n))
        n_test = n - n_train - n_val
        if n_test < 1 or n_val < 1 or n_train < 1:
            raise DatasetError(
                f"Class {config.CLASSES[class_index]} has only {n} patches; "
                "the configured split cannot give all three parts at least one."
            )
        split["train"].extend(samples[:n_train])
        split["val"].extend(samples[n_train : n_train + n_val])
        split["test"].extend(samples[n_train + n_val :])

    for part in split.values():
        rng.shuffle(part)
    return split


def optional_subset(
    samples: list[tuple[Path, int]], limit: int | None
) -> list[tuple[Path, int]]:
    """Take a stratified subset of at most ``limit`` patches, if a limit is set.

    Used only to bound a run's training time on this machine. The subset is
    drawn per class so a small budget does not quietly drop a whole class.
    """
    if limit is None or len(samples) <= limit:
        return samples

    by_class: dict[int, list[tuple[Path, int]]] = {
        i: [] for i in range(config.N_CLASSES)
    }
    for entry in samples:
        by_class[entry[1]].append(entry)

    rng = random.Random(config.SEED + 1)
    kept: list[tuple[Path, int]] = []
    # Round-robin over classes keeps the balance exact until the budget runs out.
    order = list(range(config.N_CLASSES))
    while len(kept) < limit:
        progressed = False
        for class_index in order:
            bucket = by_class[class_index]
            if bucket:
                kept.append(bucket.pop())
                progressed = True
                if len(kept) >= limit:
                    break
        if not progressed:
            break
    rng.shuffle(kept)
    return kept


def split_record(split: dict[str, list[tuple[Path, int]]]) -> dict[str, object]:
    """A JSON-serialisable summary of a split, for results/split.json."""
    record: dict[str, object] = {
        "seed": config.SEED,
        "fractions": {
            "train": config.TRAIN_FRACTION,
            "val": config.VAL_FRACTION,
            "test": config.TEST_FRACTION,
        },
        "parts": {},
    }
    parts = record["parts"]
    for name, samples in split.items():
        counts = [0] * config.N_CLASSES
        for _, class_index in samples:
            counts[class_index] += 1
        parts[name] = {
            "count": len(samples),
            "per_class": {config.CLASSES[i]: counts[i] for i in range(config.N_CLASSES)},
        }
    return record


def _recorded_path(path: Path) -> str:
    """Project relative where possible, absolute otherwise.

    The tests build their trees in a temporary directory, which is not inside
    the project, and a missing split file is worse than a long path in it.
    """
    try:
        return str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_split_json(split: dict[str, list[tuple[Path, int]]], path: Path) -> None:
    """Record the exact files used in each part, so a rerun can be checked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = split_record(split)
    payload["files"] = {
        name: [_recorded_path(p) for p, _ in samples]
        for name, samples in split.items()
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_split_json(path: Path) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("files", {})


def prepare_dataset() -> list[tuple[Path, int]]:
    """Make sure the data is on disk and return the full index."""
    if not config.IMAGES_DIR.exists() or not any(config.IMAGES_DIR.glob("*/*.jpg")):
        archive = archive_path()
        if not archive.exists():
            download_archive()
        extract_archive()
    return index_dataset()


def dataset_summary() -> dict[str, object]:
    """Counts and a hash of the archive, for the README and the console."""
    index = index_dataset()
    archive = archive_path()
    summary: dict[str, object] = {
        "total_patches": len(index),
        "classes": class_counts(index),
        "archive": {
            "name": archive.name,
            "url": config.ARCHIVE_URL,
            "size_bytes": archive.stat().st_size if archive.exists() else None,
            "sha256": file_sha256(archive) if archive.exists() else None,
        },
    }
    return summary


def write_summary_csv(summary: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = summary["classes"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "patches", "description"])
        for name, count in classes.items():
            writer.writerow([name, count, config.CLASS_DESCRIPTIONS[name]])
        writer.writerow(["total", summary["total_patches"], "all classes"])


def write_summary_json(summary: dict[str, object], path: Path) -> None:
    """The same summary as the CSV, for the figures and the console."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    """``python -m landcover.data``: fetch, extract, index and summarise."""
    prepare_dataset()
    summary = dataset_summary()
    write_summary_csv(summary, config.SUMMARY_CSV_PATH)
    write_summary_json(summary, config.SUMMARY_JSON_PATH)
    print(json.dumps(summary["classes"], indent=2))
    print(f"{summary['total_patches']} patches in {config.N_CLASSES} classes")


if __name__ == "__main__":
    main()
