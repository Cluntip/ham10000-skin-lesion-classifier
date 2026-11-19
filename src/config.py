"""Configuration loading, project paths and seeding.

Every script reads its settings from a YAML file in ``configs/`` through
:func:`load_config`, so no experiment setting or path is hard-coded.

Owner: Tech Lead.
"""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import yaml

#: Absolute path of the repository root (the folder that contains ``src/``).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: Default folder for configuration files.
CONFIG_DIR: Path = PROJECT_ROOT / "configs"


def resolve_path(path: str | os.PathLike) -> Path:
    """Turn a path from a config file into an absolute path.

    Relative paths are interpreted relative to the repository root, so the
    same config works on every machine and on Colab. POSIX-style absolute
    paths (``/tmp/x``) are treated as absolute on every OS, including
    Windows where ``Path("/tmp/x").is_absolute()`` is ``False``.

    Args:
        path: Absolute path, or path relative to the repository root.

    Returns:
        The absolute path.
    """
    path = Path(path)
    if path.is_absolute() or PurePosixPath(str(path).replace("\\", "/")).is_absolute():
        return path
    return PROJECT_ROOT / path


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``.

    Nested dictionaries are merged key by key; any other value in
    ``override`` replaces the value in ``base``.

    Args:
        base: The default values.
        override: The values that take priority.

    Returns:
        A new merged dictionary; the inputs are not modified.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_overrides(config: dict, overrides: Iterable[str]) -> dict:
    """Apply command-line overrides written as ``dotted.key=value``.

    Values are parsed as YAML, so ``0.001`` becomes a float, ``true`` a bool
    and ``[128, 128]`` a list.

    Args:
        config: The configuration to update.
        overrides: Strings such as ``"train.learning_rate=0.0001"``.

    Returns:
        A new configuration with the overrides applied.

    Raises:
        ValueError: If an override is not of the form ``key=value``.
    """
    config = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        keys = dotted_key.strip().split(".")
        node = config
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(raw_value)
    return config


def load_config(path: str | os.PathLike, overrides: Iterable[str] = ()) -> dict:
    """Load a YAML config, resolving ``extends`` and command-line overrides.

    A config may start with ``extends: base.yaml``; the parent file (relative
    to the child's folder) is loaded first and the child's keys are merged on
    top. Chains of ``extends`` are allowed.

    Args:
        path: Path to the YAML file (absolute or relative to the repo root).
        overrides: Optional ``dotted.key=value`` strings applied last.

    Returns:
        The final configuration dictionary.
    """
    path = resolve_path(path)
    with open(path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    parent = config.pop("extends", None)
    if parent is not None:
        config = deep_merge(load_config(path.parent / parent), config)

    return apply_overrides(config, overrides)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fix the random seeds of Python, NumPy and TensorFlow.

    Args:
        seed: The random seed.
        deterministic: If True, also enable TensorFlow's deterministic ops so
            two runs with the same seed on the same hardware give the same
            result (slightly slower on GPU). Set it to False if an op raises
            "does not have a deterministic GPU implementation".
    """
    import numpy as np
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
