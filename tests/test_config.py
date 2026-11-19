"""Unit tests for src/config.py (no TensorFlow needed)."""

from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, apply_overrides, deep_merge, load_config, resolve_path


def test_base_config_loads():
    config = load_config("configs/base.yaml")
    assert config["experiment"]["seed"] == 42
    assert len(config["data"]["classes"]) == 7


def test_extends_overrides_only_changed_keys():
    base = load_config("configs/base.yaml")
    child = load_config("configs/example_lr_1e-4.yaml")
    assert child["train"]["learning_rate"] == 0.0001
    assert child["train"]["batch_size"] == base["train"]["batch_size"]
    assert "extends" not in child


def test_command_line_overrides_are_typed():
    config = apply_overrides({"train": {"batch_size": 32}}, ["train.batch_size=64", "data.image_size=[128, 128]"])
    assert config["train"]["batch_size"] == 64
    assert config["data"]["image_size"] == [128, 128]


def test_bad_override_raises():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_deep_merge_does_not_modify_inputs():
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"b": 10}})
    assert merged == {"a": {"b": 10, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}


def test_resolve_path_is_relative_to_repo_root():
    assert resolve_path("data/raw") == PROJECT_ROOT / "data" / "raw"
    assert resolve_path(Path("/tmp/x")) == Path("/tmp/x")
