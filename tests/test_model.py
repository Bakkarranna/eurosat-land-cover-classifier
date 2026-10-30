"""Tests for the classifier wrapper and the checkpoint round trip."""

import pytest
import torch

from landcover import config
from landcover import model as model_module


def test_build_model_replaces_the_final_layer_for_ten_classes():
    net = model_module.build_model(pretrained=False)
    assert net.fc.out_features == config.N_CLASSES
    assert net.fc.in_features == 2048


def test_freeze_backbone_freezes_everything_except_the_head():
    net = model_module.build_model(pretrained=False)
    trainable_tensors = model_module.set_backbone_trainable(net, False)
    trainable = [name for name, p in net.named_parameters() if p.requires_grad]
    # The head is the only thing stage one trains, and the count is what the
    # training loop records.
    assert trainable == ["fc.weight", "fc.bias"]
    assert trainable_tensors == 2


def test_unfreeze_backbone_trains_every_parameter():
    net = model_module.build_model(pretrained=False)
    model_module.set_backbone_trainable(net, False)
    trainable_tensors = model_module.set_backbone_trainable(net, True)
    trainable = [name for name, p in net.named_parameters() if p.requires_grad]
    assert any(name.startswith("layer4") for name in trainable)
    assert "conv1.weight" in trainable
    assert trainable_tensors == len(trainable)


def test_forward_shape_is_one_score_per_class():
    net = model_module.build_model(pretrained=False)
    batch = torch.zeros(2, 3, config.INPUT_SIZE, config.INPUT_SIZE)
    assert tuple(net(batch).shape) == (2, config.N_CLASSES)


def test_checkpoint_round_trip(tmp_path):
    net = model_module.build_model(pretrained=False)
    path = tmp_path / "model.pt"
    saved = model_module.save_checkpoint(
        path, net, {"run": {"tag": "test"}, "epoch": 1}
    )
    reloaded, payload = model_module.load_checkpoint(path)
    assert payload["epoch"] == 1
    assert payload["run"] == {"tag": "test"}
    # The architecture block is what lets a checkpoint be paired back to the
    # network that produced it, so it has to survive the save.
    assert saved["architecture"] == payload["architecture"]
    assert saved["architecture"]["n_classes"] == config.N_CLASSES
    batch = torch.ones(1, 3, config.INPUT_SIZE, config.INPUT_SIZE)
    with torch.no_grad():
        assert torch.allclose(net(batch), reloaded(batch))


def test_load_checkpoint_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        model_module.load_checkpoint(tmp_path / "absent.pt")
