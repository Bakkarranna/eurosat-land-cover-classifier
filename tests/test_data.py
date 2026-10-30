"""Tests for dataset indexing, splitting, and the transforms."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from landcover import config, data
from landcover.dataset import LandcoverDataset, build_eval_transform


def _write_patch(path, color=(10, 120, 30)):
    # The real patches are 64x64; the transform is what resizes them to 224.
    Image.new("RGB", (config.PATCH_SIZE, config.PATCH_SIZE), color).save(path, "JPEG")


def _fake_tree(tmp_path, per_class=4) -> Path:
    """Write ``per_class`` JPEGs into each of the ten class folders."""
    root = tmp_path / "EuroSAT"
    for index in range(config.N_CLASSES):
        class_dir = root / config.CLASSES[index]
        class_dir.mkdir(parents=True)
        for j in range(per_class):
            _write_patch(class_dir / f"patch_{j}.jpg", color=(index * 20, 80, 60))
    return root


def _index(root):
    return [(path, config.CLASSES.index(path.parent.name)) for path in sorted(root.glob("*/*.jpg"))]


def test_index_dataset_finds_every_patch_in_a_stable_order(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGES_DIR", _fake_tree(tmp_path))
    index = data.index_dataset()
    assert len(index) == 4 * config.N_CLASSES
    # Same input, same order: what makes the seeded split reproducible.
    assert index == data.index_dataset()
    assert all(path.parent.name == config.CLASSES[i] for path, i in index)


def test_class_counts_are_reported_in_class_order(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGES_DIR", _fake_tree(tmp_path))
    counts = data.class_counts(data.index_dataset())
    assert list(counts) == config.CLASSES
    assert set(counts.values()) == {4}


def test_index_dataset_rejects_a_missing_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "Nothing")
    with pytest.raises(data.DatasetError):
        data.index_dataset()


def test_index_dataset_rejects_a_partial_tree(tmp_path, monkeypatch):
    root = _fake_tree(tmp_path)
    for patch in (root / "Forest").glob("*.jpg"):
        patch.unlink()
    monkeypatch.setattr(config, "IMAGES_DIR", root)
    with pytest.raises(data.DatasetError) as error:
        data.index_dataset()
    assert "Forest" in str(error.value)


def test_split_dataset_is_stratified_disjoint_and_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGES_DIR", _fake_tree(tmp_path, per_class=10))
    index = data.index_dataset()

    first = data.split_dataset(index, seed=0)
    second = data.split_dataset(index, seed=0)
    other = data.split_dataset(index, seed=1)
    assert first == second
    assert first != other

    assert len(first["train"]) == 8 * config.N_CLASSES
    assert len(first["val"]) == config.N_CLASSES
    assert len(first["test"]) == config.N_CLASSES
    for part in ("train", "val", "test"):
        counts = [0] * config.N_CLASSES
        for _, class_index in first[part]:
            counts[class_index] += 1
        assert counts == [len(first[part]) // config.N_CLASSES] * config.N_CLASSES

    seen = [p for part in first.values() for p, _ in part]
    assert len(seen) == len(set(seen)) == len(index)


def test_split_dataset_rejects_a_class_too_small(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGES_DIR", _fake_tree(tmp_path, per_class=2))
    with pytest.raises(data.DatasetError):
        data.split_dataset(data.index_dataset(), seed=0)


def test_optional_subset_keeps_every_class(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGES_DIR", _fake_tree(tmp_path, per_class=10))
    index = data.index_dataset()
    subset = data.optional_subset(index, limit=25)
    assert len(subset) == 25
    assert len({class_index for _, class_index in subset}) == config.N_CLASSES


def test_dataset_returns_a_float_tensor_and_an_integer_label(tmp_path):
    path = tmp_path / "patch.jpg"
    _write_patch(path)
    # LandcoverDataset takes (path, class_index) pairs, the same shape
    # index_dataset and split_dataset produce.
    image, label = LandcoverDataset([(path, 3)], transform=build_eval_transform())[0]
    assert image.shape == (3, config.INPUT_SIZE, config.INPUT_SIZE)
    assert image.dtype.is_floating_point
    assert label == 3


def test_build_eval_transform_is_deterministic():
    transform = build_eval_transform()
    image = Image.new("RGB", (config.PATCH_SIZE, config.PATCH_SIZE), (10, 120, 30))
    assert np.array_equal(np.asarray(transform(image)), np.asarray(transform(image)))


def test_split_json_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGES_DIR", _fake_tree(tmp_path, per_class=10))
    split = data.split_dataset(data.index_dataset(), seed=0)
    path = tmp_path / "split.json"
    data.write_split_json(split, path)
    record = data.read_split_json(path)
    assert set(record) == {"train", "val", "test"}
    assert len(record["train"]) == len(split["train"])
