"""Inference on a raw image file.

Loads a saved model and the YAML settings used to train it.

Usage::

    python -m src.predict path/to/image.jpg

Owner: Evaluation & Deployment Engineer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config, resolve_path
from src.data import load_image


def predict_image(
    image_path: str | Path,
    *,
    config_path: str | Path = "configs/base.yaml",
    model_path: str | Path = "models/final.keras",
) -> dict:
    """Predict the lesion class of one image.

    Args:
        image_path: Path to a JPEG or PNG dermatoscopic image.
        config_path: YAML configuration used to train the saved model.
        model_path: Path to the saved Keras classifier.

    Returns:
        ``{"label": str, "confidence": float, "probabilities": {class: prob}}``.
    """
    from keras.models import load_model

    config = load_config(config_path)
    classes = config["data"]["classes"]
    model_file = resolve_path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Saved Keras model not found at {model_file}")

    model = load_model(model_file, compile=False)
    image = load_image(str(image_path), config)
    scores = model(image[None, ...], training=False)
    if tuple(scores.shape) != (1, len(classes)):
        raise ValueError(f"Expected one model score per YAML class, got shape {scores.shape}")

    probabilities = dict(zip(classes, map(float, scores[0]), strict=True))
    label = max(probabilities, key=probabilities.__getitem__)
    return {"label": label, "confidence": probabilities[label], "probabilities": probabilities}


def main() -> None:
    """Command-line entry point: print the prediction for one image."""
    parser = argparse.ArgumentParser(description="Predict the class of a dermatoscopic image.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--model", type=Path, default=Path("models/final.keras"))
    args = parser.parse_args()
    result = predict_image(args.image_path, config_path=args.config, model_path=args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
