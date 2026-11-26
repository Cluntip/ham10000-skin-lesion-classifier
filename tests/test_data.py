"""Tests for HAM10000 metadata loading and lesion-level splitting."""

from pathlib import Path

import pandas as pd
import pytest

from src.config import PROJECT_ROOT
from src.data import load_metadata, split_by_lesion


def make_config(tmp_path: Path) -> dict:
    return {
        "experiment": {"seed": 42},
        "data": {
            "raw_dir": str(tmp_path),
            "metadata_file": "HAM10000_metadata.csv",
            "classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
            "split": {"train": 0.70, "val": 0.15, "test": 0.15},
        },
    }


def test_load_metadata_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="metadata not found"):
        load_metadata(make_config(tmp_path))


def test_load_metadata_attaches_discovered_image_paths(tmp_path: Path):
    image_dir = tmp_path / "HAM10000_images"
    image_dir.mkdir()
    image_path = image_dir / "ISIC_001.jpg"
    image_path.touch()
    pd.DataFrame(
        {
            "image_id": ["ISIC_001"],
            "lesion_id": ["lesion_001"],
            "dx": ["mel"],
        }
    ).to_csv(tmp_path / "HAM10000_metadata.csv", index=False)

    metadata = load_metadata(make_config(tmp_path))

    assert metadata.loc[0, "image_path"] == image_path


def test_split_by_lesion_has_no_cross_split_leakage():
    rows = []
    for lesion_number in range(30):
        lesion_id = f"lesion_{lesion_number:02d}"
        diagnosis = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"][
            lesion_number % 7
        ]
        for image_number in range(2):
            rows.append(
                {
                    "image_id": f"image_{lesion_number:02d}_{image_number}",
                    "lesion_id": lesion_id,
                    "dx": diagnosis,
                }
            )
    metadata = pd.DataFrame(rows)
    config = make_config(PROJECT_ROOT / "data" / "raw")

    splits = split_by_lesion(metadata, config)

    lesion_sets = [
        set(splits[name]["lesion_id"].astype(str))
        for name in ("train", "val", "test")
    ]
    assert len(splits["train"]) > len(splits["val"])
    assert len(splits["train"]) > len(splits["test"])
    assert lesion_sets[0].isdisjoint(lesion_sets[1])
    assert lesion_sets[0].isdisjoint(lesion_sets[2])
    assert lesion_sets[1].isdisjoint(lesion_sets[2])
    assert set().union(*lesion_sets) == set(metadata["lesion_id"].astype(str))

def test_split_follows_configured_ratios_and_fixed_holdout_seed(tmp_path):
    config = make_config(tmp_path)
    config["data"]["split"].update(train=0.60, val=0.25, test=0.15, seed=42)
    metadata = pd.DataFrame(
        [
            {"image_id": f"{dx}_{index}", "lesion_id": f"{dx}_{index}", "dx": dx}
            for dx in config["data"]["classes"]
            for index in range(20)
        ]
    )

    splits = split_by_lesion(metadata, config)
    config["experiment"]["seed"] = 99
    repeated = split_by_lesion(metadata, config)

    for name, count in {"train": 12, "val": 5, "test": 3}.items():
        assert splits[name]["dx"].value_counts().to_dict() == dict.fromkeys(
            config["data"]["classes"], count
        )
        pd.testing.assert_frame_equal(splits[name], repeated[name])


def test_class_weights_use_only_training_labels(tmp_path):
    from src.data import compute_class_weights

    config = make_config(tmp_path)
    rows = []
    for index, dx in enumerate(config["data"]["classes"]):
        for number in range(3 + index):
            image_id = f"{dx}_{number}"
            (tmp_path / f"{image_id}.jpg").touch()
            rows.append({"image_id": image_id, "lesion_id": image_id, "dx": dx})
    pd.DataFrame(rows + rows[:1]).to_csv(tmp_path / "HAM10000_metadata.csv", index=False)

    train = split_by_lesion(pd.DataFrame(rows), config)["train"]
    counts = train["dx"].value_counts()
    expected = {
        index: len(train) / (len(config["data"]["classes"]) * counts[dx])
        for index, dx in enumerate(config["data"]["classes"])
    }
    assert compute_class_weights(config) == pytest.approx(expected)


def test_tensorflow_pipeline_augments_only_training_images(tmp_path):
    import numpy as np
    from PIL import Image

    from src.data import make_datasets

    pytest.importorskip("tensorflow")
    config = make_config(tmp_path)
    config["data"].update(image_size=[12, 16], augmentation=False)
    config["train"] = {"batch_size": 32}
    pixels = np.zeros((6, 8, 3), dtype=np.uint8)
    pixels[:3, :4, 0] = 255
    rows = []
    for dx in config["data"]["classes"]:
        for number in range(4):
            image_id = f"{dx}_{number}"
            Image.fromarray(pixels).save(tmp_path / f"{image_id}.png")
            rows.append({"image_id": image_id, "lesion_id": image_id, "dx": dx})
    pd.DataFrame(rows).to_csv(tmp_path / "HAM10000_metadata.csv", index=False)

    baseline = make_datasets(config)
    config["data"]["augmentation"] = True
    augmented = make_datasets(config)
    for name in ("train", "val", "test"):
        images, labels = next(baseline[name].as_numpy_iterator())
        changed_images, changed_labels = next(augmented[name].as_numpy_iterator())
        assert images.shape[1:] == (12, 16, 3)
        assert images.dtype == np.float32
        assert images.min() == 0 and images.max() == 1
        assert labels.shape[1:] == (7,)
        assert np.all(labels.sum(axis=1) == 1)
        np.testing.assert_array_equal(labels, changed_labels)
        if name == "train":
            assert not np.array_equal(images, changed_images)
        else:
            np.testing.assert_array_equal(images, changed_images)


def test_audit_exports_only_after_image_validation(tmp_path):
    from PIL import Image

    from src.data import audit

    config = make_config(tmp_path)
    config["data"]["processed_dir"] = str(tmp_path / "processed")
    rows = []
    for dx in config["data"]["classes"]:
        for number in range(3):
            image_id = f"{dx}_{number}"
            Image.new("RGB", (2, 2)).save(tmp_path / f"{image_id}.jpg")
            rows.append({"image_id": image_id, "lesion_id": image_id, "dx": dx})
    pd.DataFrame(rows).to_csv(tmp_path / "HAM10000_metadata.csv", index=False)

    manifest = tmp_path / "processed" / "ham10000_split_seed42.csv"
    corrupt_image = tmp_path / "nv_0.jpg"
    corrupt_image.write_bytes(b"invalid JPEG")
    with pytest.raises(ValueError, match="1 unreadable images"):
        audit(config, save_split=True)
    assert not manifest.exists()

    Image.new("RGB", (2, 2)).save(corrupt_image)
    audit(config, save_split=True)
    assignments = pd.read_csv(manifest)
    assert list(assignments.columns) == ["image_id", "lesion_id", "dx", "split"]
    assert assignments["image_id"].is_unique
    assert set(assignments["image_id"]) == {row["image_id"] for row in rows}
    assert set(assignments["split"]) == {"train", "val", "test"}
