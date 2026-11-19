"""Inference on a raw image file.

Loads the saved model and preprocessing settings from ``models/`` and applies
exactly the same preprocessing as training (``src.data.preprocess_image``).

Usage::

    python -m src.predict path/to/image.jpg

Owner: Evaluation & Deployment Engineer. The functions below are the agreed
interface; their bodies are to be implemented.
"""

from __future__ import annotations

from pathlib import Path


def predict_image(image_path: str | Path) -> dict:
    """Predict the lesion class of one image.

    Args:
        image_path: Path to a JPEG or PNG dermatoscopic image.

    Returns:
        ``{"label": str, "confidence": float, "probabilities": {class: prob}}``.
    """
    raise NotImplementedError


def main() -> None:
    """Command-line entry point: print the prediction for one image."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
