"""Baselines and neural architectures.

``build_model`` picks the builder named by ``config["model"]["name"]``, so
a new architecture only needs a new function and one entry in
``MODEL_BUILDERS``.

Owners: Tech Lead (baselines and Model A), Model Architect (Models B and C).
The builders below are the agreed interface; their bodies are to be
implemented.
"""

from __future__ import annotations

from typing import Callable

import keras


def build_custom_cnn(config: dict) -> keras.Model:
    """Model A: a CNN trained from scratch.

    Args:
        config: The project configuration.

    Returns:
        An uncompiled Keras model with a 7-way softmax output.
    """
    raise NotImplementedError


def build_resnet50_frozen(config: dict) -> keras.Model:
    """Model B: ImageNet ResNet-50 with a frozen base and a new classifier.

    Args:
        config: The project configuration.

    Returns:
        An uncompiled Keras model with a 7-way softmax output.
    """
    raise NotImplementedError


def build_resnet50_finetune(config: dict) -> keras.Model:
    """Model C: ResNet-50 with its last block unfrozen (small learning rate).

    Args:
        config: The project configuration.

    Returns:
        An uncompiled Keras model with a 7-way softmax output.
    """
    raise NotImplementedError


MODEL_BUILDERS: dict[str, Callable[[dict], keras.Model]] = {
    "custom_cnn": build_custom_cnn,
    "resnet50_frozen": build_resnet50_frozen,
    "resnet50_finetune": build_resnet50_finetune,
}


def build_model(config: dict) -> keras.Model:
    """Build the neural model named in ``config["model"]["name"]``.

    Args:
        config: The project configuration.

    Returns:
        An uncompiled Keras model.

    Raises:
        ValueError: If the model name is not in ``MODEL_BUILDERS``.
    """
    name = config["model"]["name"]
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_BUILDERS)}.")
    return MODEL_BUILDERS[name](config)
