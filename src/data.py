"""Data loading, cleaning, splitting and preprocessing for HAM10000.

All preprocessing is defined here once and reused by training
(``src/train.py``) and inference (``src/predict.py``), so the two can never
drift apart. Every scaler or encoder is fitted on the training split only.

Owner: Data Engineer. The functions below are the agreed interface; their
bodies are to be implemented.
"""

from __future__ import annotations

import pandas as pd
import tensorflow as tf


def load_metadata(config: dict) -> pd.DataFrame:
    """Read ``HAM10000_metadata.csv`` and attach the path of each image.

    Args:
        config: The project configuration.

    Returns:
        One row per image, with at least ``image_id``, ``lesion_id``, ``dx``
        and ``image_path`` columns.
    """
    raise NotImplementedError


def split_by_lesion(metadata: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """Split into train / val / test so that each lesion is in one split only.

    The split is grouped by ``lesion_id`` (no leakage between images of the
    same lesion) and stratified by ``dx`` as closely as possible.

    Args:
        metadata: Output of :func:`load_metadata`.
        config: The project configuration (``data.split`` section).

    Returns:
        A dictionary with keys ``"train"``, ``"val"`` and ``"test"``.
    """
    raise NotImplementedError


def preprocess_image(image: tf.Tensor, config: dict) -> tf.Tensor:
    """Resize and scale one decoded RGB image.

    Used by both training and inference.

    Args:
        image: A ``uint8`` tensor of shape ``(height, width, 3)``.
        config: The project configuration.

    Returns:
        A ``float32`` tensor of shape ``(*data.image_size, 3)``.
    """
    raise NotImplementedError


def make_datasets(config: dict) -> dict[str, tf.data.Dataset]:
    """Build batched ``tf.data`` pipelines for every split.

    Augmentation (if enabled) is applied to the training split only.

    Args:
        config: The project configuration.

    Returns:
        A dictionary with keys ``"train"``, ``"val"`` and ``"test"``; each
        yields ``(images, one_hot_labels)`` batches.
    """
    raise NotImplementedError


def compute_class_weights(config: dict) -> dict[int, float]:
    """Compute class weights from the training split's label counts.

    Args:
        config: The project configuration.

    Returns:
        A mapping ``class_index -> weight`` for ``model.fit(class_weight=...)``.
    """
    raise NotImplementedError
