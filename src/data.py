"""Data loading, cleaning, splitting and preprocessing for HAM10000.

Training and inference share the image decoder and preprocessing. Class
indices follow the configured class order.

Owner: Data Engineer.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from PIL import Image

from src.config import load_config, resolve_path

if TYPE_CHECKING:
    import tensorflow as tf


CLASS_COLUMNS = ("image_id", "lesion_id", "dx")


def load_metadata(config: dict) -> pd.DataFrame:
    """Read ``HAM10000_metadata.csv`` and attach the path of each image.

    Args:
        config: The project configuration.

    Returns:
        One row per image, with at least ``image_id``, ``lesion_id``, ``dx``
        and ``image_path`` columns.
    """
    data_cfg = config["data"]
    raw_dir = resolve_path(data_cfg["raw_dir"])
    metadata_path = resolve_path(raw_dir / data_cfg["metadata_file"])
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"HAM10000 metadata not found at {metadata_path}. "
            "The dataset must be downloaded and extracted before loading it."
        )

    metadata = pd.read_csv(metadata_path, dtype={column: "string" for column in CLASS_COLUMNS})
    missing_columns = [column for column in CLASS_COLUMNS if column not in metadata.columns]
    if missing_columns:
        raise ValueError(
            "HAM10000 metadata is missing required columns: "
            + ", ".join(missing_columns)
        )

    image_paths = {}
    for image_path in sorted(raw_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            if image_path.stem in image_paths:
                raise ValueError(f"Ambiguous image ID {image_path.stem}: "
                                 f"{image_paths[image_path.stem]} and {image_path}")
            image_paths[image_path.stem] = image_path

    image_ids = metadata["image_id"].str.strip()
    metadata["image_path"] = image_ids.map(image_paths)
    metadata.attrs["files_found"] = len(image_paths)
    metadata.attrs["unmatched_files"] = sorted(set(image_paths) - set(image_ids))
    return metadata


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
    required = {"image_id", "lesion_id", "dx"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {', '.join(missing)}")
    if metadata[list(required)].isna().any().any():
        raise ValueError("Metadata contains missing image_id, lesion_id, or dx values.")
    if metadata["image_id"].duplicated().any():
        duplicate_ids = metadata.loc[
            metadata["image_id"].duplicated(keep=False), "image_id"
        ].unique().tolist()
        raise ValueError(
            "Duplicate image_id values make a safe split ambiguous: "
            + ", ".join(map(str, duplicate_ids[:10]))
        )

    names = ("train", "val", "test")
    ratios = np.array([config["data"]["split"][name] for name in names], dtype=float)
    if not np.isfinite(ratios).all() or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1):
        raise ValueError("Split ratios must be positive and sum to 1.")

    classes = config["data"]["classes"]
    unknown = sorted(set(metadata["dx"]) - set(classes))
    if unknown:
        raise ValueError(f"Unexpected diagnosis labels: {unknown}")

    # Stratify independently within each diagnosis, allocating whole lesions
    # against target image counts. Larger lesions are placed first.
    lesions = metadata.groupby("lesion_id", sort=True)["dx"].agg(["first", "size", "nunique"])
    if lesions["nunique"].ne(1).any():
        raise ValueError("At least one lesion has conflicting diagnosis labels.")
    rng = np.random.default_rng(config["data"]["split"].get("seed", config["experiment"]["seed"]))
    assignments = {}
    total_counts = np.zeros(3)
    for _, group in lesions.groupby("first", sort=True):
        group = group.iloc[rng.permutation(len(group))]
        group = group.sort_values("size", ascending=False, kind="stable")
        target = ratios * group["size"].sum()
        counts = np.zeros(3)
        # Keep at least one lesion of each observed class in training and
        # reserve one for each holdout whenever the class has enough lesions.
        for index, (lesion, size) in enumerate(group["size"].items()):
            empty = np.flatnonzero(counts == 0)
            if index == 0:
                split = 0
            elif len(group) >= 3 and len(group) - index == len(empty):
                priority = np.lexsort((total_counts[empty] / ratios[empty], -ratios[empty]))
                split = empty[priority[0]]
            else:
                # Minimize the increase in squared error relative to target size.
                cost = (2 * counts + size) / target
                split = np.lexsort((total_counts / ratios, cost))[0]
            counts[split] += size
            total_counts[split] += size
            assignments[lesion] = names[split]

    membership = metadata["lesion_id"].map(assignments)
    splits = {
        name: metadata.loc[membership.eq(name)].reset_index(drop=True)
        for name in names
    }
    if any(frame.empty for frame in splits.values()):
        raise ValueError("Insufficient lesion groups for three nonempty stratified splits.")
    for column in ("lesion_id", "image_id"):
        groups = [set(frame[column]) for frame in splits.values()]
        if sum(map(len, groups)) != len(set().union(*groups)):
            raise RuntimeError(f"Split leakage detected for {column}.")
    if sum(map(len, splits.values())) != len(metadata):
        raise RuntimeError("Split assignments omitted metadata rows.")

    return splits


def preprocess_image(image: tf.Tensor, config: dict) -> tf.Tensor:
    """Resize and scale one decoded RGB image.

    Used by both training and inference.

    Args:
        image: A ``uint8`` tensor of shape ``(height, width, 3)``.
        config: The project configuration.

    Returns:
        A ``float32`` tensor of shape ``(*data.image_size, 3)``.
    """
    import tensorflow as tf

    image = tf.ensure_shape(image, (None, None, 3))
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.resize(image, config["data"]["image_size"])


def load_image(path: str | tf.Tensor, config: dict) -> tf.Tensor:
    """Read an image as RGB and apply the configured base preprocessing."""
    import tensorflow as tf

    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return preprocess_image(image, config)


def make_datasets(config: dict) -> dict[str, tf.data.Dataset]:
    """Build batched ``tf.data`` pipelines for every split.

    Augmentation (if enabled) is applied to the training split only.

    Args:
        config: The project configuration.

    Returns:
        A dictionary with keys ``"train"``, ``"val"`` and ``"test"``; each
        yields ``(images, one_hot_labels)`` batches.
    """
    import tensorflow as tf

    data_cfg = config["data"]
    classes = data_cfg["classes"]
    class_ids = {name: index for index, name in enumerate(classes)}
    seed = config["experiment"]["seed"]
    batch_size = config["train"]["batch_size"]

    def decode(path, label):
        return load_image(path, config), tf.one_hot(label, len(classes))

    datasets = {}
    for name, frame in _prepared_splits(config).items():
        paths = frame["image_path"].map(str).tolist()
        labels = frame["dx"].map(class_ids).to_numpy(dtype=np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        if name == "train":
            dataset = dataset.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
        if name == "train" and data_cfg["augmentation"]:
            augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal", seed=seed),
                tf.keras.layers.RandomRotation(0.025, fill_mode="reflect", seed=seed + 1),
            ])
            dataset = dataset.map(lambda images, labels: (augmentation(images, training=True), labels))
        datasets[name] = dataset.prefetch(tf.data.AUTOTUNE)
    return datasets


def compute_class_weights(config: dict) -> dict[int, float]:
    """Compute class weights from the training split's label counts.

    Args:
        config: The project configuration.

    Returns:
        A mapping ``class_index -> weight`` for ``model.fit(class_weight=...)``.
    """
    return _class_weights(_prepared_splits(config)["train"], config)


def _class_weights(train: pd.DataFrame, config: dict) -> dict[int, float]:
    classes = config["data"]["classes"]
    counts = train["dx"].value_counts().reindex(classes, fill_value=0)
    if counts.eq(0).any():
        raise ValueError(f"Training split lacks diagnosis labels: {counts[counts.eq(0)].index.tolist()}")
    return {
        index: len(train) / (len(classes) * count)
        for index, count in enumerate(counts)
    }


def _prepare_metadata(metadata: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop exact repeated rows and stop on ambiguous or incomplete records."""
    cleaned = metadata.drop_duplicates().copy()
    for column in CLASS_COLUMNS:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        invalid = cleaned[column].isna() | cleaned[column].eq("")
        if invalid.any():
            raise ValueError(f"Missing {column} at CSV row indices {cleaned.index[invalid].tolist()[:10]}.")
    missing = cleaned.loc[cleaned["image_path"].isna(), "image_id"]
    if not missing.empty:
        raise FileNotFoundError(f"{len(missing)} metadata records lack images: {missing.head(10).tolist()}")
    absent = set(config["data"]["classes"]) - set(cleaned["dx"])
    if absent:
        raise ValueError(f"Dataset lacks configured diagnosis labels: {sorted(absent)}")
    return cleaned


def _prepared_splits(config: dict) -> dict[str, pd.DataFrame]:
    return split_by_lesion(_prepare_metadata(load_metadata(config), config), config)


def audit(config: dict, *, save_split: bool = False) -> None:
    """Inspect and decode the dataset, optionally exporting validated split assignments."""
    metadata = load_metadata(config)
    counts = metadata["dx"].value_counts(dropna=False)
    print(f"Metadata shape: {metadata.shape}; column types: {metadata.dtypes.to_dict()}")
    print(metadata.head().drop(columns="image_path").to_string(index=False))
    print(f"Missing values: {metadata.isna().sum().to_dict()}")
    print(f"Duplicate rows: {metadata.duplicated().sum()}; duplicate image IDs: "
          f"{metadata['image_id'].duplicated().sum()}")
    print(f"Diagnosis distribution: {pd.DataFrame({'count': counts, 'percent': 100 * counts / len(metadata)})}")
    print(f"Lesions: {metadata['lesion_id'].nunique()}; images per lesion: "
          f"{metadata.groupby('lesion_id').size().describe().to_dict()}")
    missing = metadata.loc[metadata["image_path"].isna(), "image_id"]
    orphaned = metadata.attrs["unmatched_files"]
    print(f"Image files: {metadata.attrs['files_found']}; matched rows: {len(metadata) - len(missing)}; "
          f"unmatched rows: {len(missing)} {missing.head(10).tolist()}; "
          f"unmatched files: {len(orphaned)} {orphaned[:10]}")

    image_types = Counter()
    unreadable = []
    for path in metadata["image_path"].dropna().unique():
        try:
            with Image.open(path) as image:
                image.load()
                image_types[(image.size, image.mode, image.format)] += 1
        except (OSError, ValueError) as error:
            unreadable.append((str(path), str(error)))
    print(f"Image sizes, modes and formats: {dict(image_types)}")
    if unreadable:
        raise ValueError(f"{len(unreadable)} unreadable images: {unreadable}")

    cleaned = _prepare_metadata(metadata, config)
    splits = split_by_lesion(cleaned, config)
    print(f"Exact duplicate rows removed: {len(metadata) - len(cleaned)}")
    for name, frame in splits.items():
        counts = frame["dx"].value_counts().reindex(config["data"]["classes"], fill_value=0)
        print(f"{name}: {len(frame)} images ({len(frame) / len(cleaned):.1%}), "
              f"{frame['lesion_id'].nunique()} lesions, diagnoses: {counts.to_dict()}")
    print("Image and lesion IDs are disjoint across splits.")
    weights = _class_weights(splits["train"], config)
    print(f"Training class weights: {weights}")
    if save_split:
        print(f"Saved split assignments to {_save_split_assignments(splits, config)}")


def _save_split_assignments(splits: dict[str, pd.DataFrame], config: dict) -> Path:
    data_cfg = config["data"]
    seed = data_cfg["split"].get("seed", config["experiment"]["seed"])
    destination = resolve_path(data_cfg["processed_dir"]) / f"ham10000_split_seed{seed}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    assignments = pd.concat([
        frame[list(CLASS_COLUMNS)].assign(split=name)
        for name, frame in splits.items()
    ], ignore_index=True)
    temporary = NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=destination.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            assignments.to_csv(temporary, index=False)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit HAM10000 files and split assignments.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--save-split", action="store_true", help="Export the split after a successful audit.")
    args = parser.parse_args()
    audit(load_config(args.config, args.set), save_split=args.save_split)
