"""Inference on a raw image file.

Loads the saved model and its inference config from ``models/`` and applies
exactly the same image reading and preprocessing as training
(``src.data.load_image`` / ``src.data.preprocess_image``), so training and
inference can never drift apart.

Artefacts in ``models/`` (written by :func:`save_artifacts`):

* ``final_model.keras``      the trained Keras model (softmax over the classes)
* ``inference_config.json``  the full config the model was trained with; it
  holds everything preprocessing needs (``data.image_size``, ``data.classes``)

Usage::

    python -m src.predict path/to/image.jpg [more.jpg ...]
    python -m src.predict path/to/image.jpg --json
    python -m src.predict path/to/image.jpg --benchmark 50     # inference time and model size

Owner: Evaluation & Deployment Engineer.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from src.config import PROJECT_ROOT

#: Default folder for the final model and its preprocessing settings.
DEFAULT_MODEL_DIR: Path = PROJECT_ROOT / "models"
MODEL_FILE = "final_model.keras"
CONFIG_FILE = "inference_config.json"

#: Classes that are malignant or pre-malignant (same list as src.evaluate).
MALIGNANT_CLASSES: tuple[str, ...] = ("akiec", "bcc", "mel")


# ---------------------------------------------------------------------------
# Saving and loading the artefacts
# ---------------------------------------------------------------------------


def save_artifacts(model, config: dict, model_dir: str | Path | None = None) -> Path:
    """Save the final model and the config that inference needs.

    Called once by whoever trains the final model (Tech Lead), so that
    :func:`predict_image` can rebuild the exact preprocessing.

    Args:
        model: The trained Keras model.
        config: The configuration the model was trained with.
        model_dir: Output folder (default ``models/``).

    Returns:
        The folder the artefacts were written to.
    """
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / MODEL_FILE)
    with open(model_dir / CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return model_dir


@functools.lru_cache(maxsize=4)
def load_artifacts(model_dir: str | Path | None = None) -> tuple:
    """Load the model and its inference config (cached, so it is loaded once).

    Args:
        model_dir: Folder with the artefacts (default ``models/``).

    Returns:
        ``(model, config)``.

    Raises:
        FileNotFoundError: If the model or config file is missing.
    """
    import keras

    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    model_path, config_path = model_dir / MODEL_FILE, model_dir / CONFIG_FILE
    for path in (model_path, config_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. See models/README.md for how to get the final model.")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return keras.models.load_model(model_path), config


@functools.lru_cache(maxsize=4)
def compiled_forward(model_dir: str | Path | None = None):
    """Return the model's forward pass compiled into a TensorFlow graph.

    Calling a Keras model eagerly (``model(x)``) runs every layer as a
    separate Python-level operation, which for ResNet-50 is much slower than
    running one compiled graph. ``tf.function`` traces the model once; the
    fixed ``input_signature`` (one image of the training size) means it is
    never re-traced, whatever the size of the uploaded image (that is handled
    by the preprocessing before this step).

    Args:
        model_dir: Folder with the artefacts (default ``models/``).

    Returns:
        A function mapping a ``float32`` batch ``(1, h, w, 3)`` to class
        probabilities ``(1, n_classes)``.
    """
    import tensorflow as tf

    model, config = load_artifacts(model_dir)
    height, width = config["data"]["image_size"]

    @tf.function(input_signature=[tf.TensorSpec((1, height, width, 3), tf.float32)])
    def forward(batch):
        return model(batch, training=False)

    return forward


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _predict_preprocessed(x, config: dict, model_dir, start: float) -> dict:
    """Run the compiled model on one preprocessed image and format the result.

    Args:
        x: ``float32`` tensor of shape ``(*data.image_size, 3)``.
        config: The inference config.
        model_dir: Folder with the artefacts.
        start: ``time.perf_counter()`` value when the request started.

    Returns:
        See :func:`predict_array`.
    """
    import tensorflow as tf

    class_names = config["data"]["classes"]
    probs = compiled_forward(model_dir)(tf.expand_dims(tf.cast(x, tf.float32), 0)).numpy()[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    order = np.argsort(-probs)
    best = int(order[0])
    return {
        "label": class_names[best],
        "confidence": float(probs[best]),
        "probabilities": {class_names[i]: float(probs[i]) for i in order},
        "malignant_probability": float(sum(probs[i] for i, c in enumerate(class_names) if c in MALIGNANT_CLASSES)),
        "inference_ms": elapsed_ms,
    }


def predict_array(image, model_dir: str | Path | None = None) -> dict:
    """Predict the lesion class of one decoded RGB image.

    Used by the demo app (Gradio passes a NumPy array). The image goes
    through ``src.data.preprocess_image``, the same function used in training.

    Args:
        image: ``uint8`` array or tensor of shape ``(h, w, 3)``.
        model_dir: Folder with the artefacts (default ``models/``).

    Returns:
        ``{"label", "confidence", "probabilities", "malignant_probability",
        "inference_ms"}``; ``probabilities`` maps every class to its
        probability, sorted from most to least likely.
    """
    import tensorflow as tf

    from src import data

    _, config = load_artifacts(model_dir)
    start = time.perf_counter()
    x = data.preprocess_image(tf.convert_to_tensor(image, dtype=tf.uint8), config)
    return _predict_preprocessed(x, config, model_dir, start)


def predict_image(image_path: str | Path, model_dir: str | Path | None = None) -> dict:
    """Predict the lesion class of one image file.

    The file is read with ``src.data.load_image``, the exact function the
    training pipeline uses (same decoder, same preprocessing).

    Args:
        image_path: Path to a JPEG or PNG dermatoscopic image.
        model_dir: Folder with the artefacts (default ``models/``).

    Returns:
        ``{"label": str, "confidence": float, "probabilities": {class: prob},
        "malignant_probability": float, "inference_ms": float}``.
    """
    from src import data

    _, config = load_artifacts(model_dir)
    start = time.perf_counter()
    x = data.load_image(str(image_path), config)
    return _predict_preprocessed(x, config, model_dir, start)


# ---------------------------------------------------------------------------
# Deployment measurements (spec section 3.7)
# ---------------------------------------------------------------------------


def warm_up(model_dir: str | Path | None = None) -> float:
    """Load the model and build its graph before the first real request.

    The first prediction is slow (several seconds) because the model is
    loaded and :func:`compiled_forward` traces the graph. The demo calls this
    at start-up so the first user does not wait.

    Args:
        model_dir: Folder with the artefacts (default ``models/``).

    Returns:
        How long the warm-up took, in milliseconds.
    """
    start = time.perf_counter()
    predict_array(np.zeros((64, 64, 3), dtype=np.uint8), model_dir)
    return (time.perf_counter() - start) * 1000


def model_size_mb(model_dir: str | Path | None = None) -> float:
    """Size of the saved model file on disk, in megabytes."""
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    return (model_dir / MODEL_FILE).stat().st_size / 1e6


def benchmark(image_path: str | Path, n_runs: int = 50, warmup: int = 5, model_dir: str | Path | None = None) -> dict:
    """Measure the inference time per sample on this machine.

    The first calls are slower (graph tracing, GPU warm-up), so ``warmup``
    runs are discarded before timing. Image decoding is included, because a
    user of the demo waits for it too.

    Args:
        image_path: Any image to predict repeatedly.
        n_runs: Number of timed predictions.
        warmup: Number of untimed predictions first.
        model_dir: Folder with the artefacts (default ``models/``).

    Returns:
        ``{"mean_ms", "std_ms", "median_ms", "n_runs", "model_size_mb", "device"}``.
    """
    import tensorflow as tf

    for _ in range(warmup):
        predict_image(image_path, model_dir)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_image(image_path, model_dir)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "n_runs": n_runs,
        "model_size_mb": model_size_mb(model_dir),
        "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Argument list; ``None`` means ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Predict the skin-lesion class of one or more images.")
    parser.add_argument("images", nargs="+", help="Paths to JPEG / PNG images.")
    parser.add_argument("--model-dir", default=None, help="Folder with final_model.keras and inference_config.json.")
    parser.add_argument("--json", action="store_true", help="Print the raw result as JSON.")
    parser.add_argument("--benchmark", type=int, default=0, metavar="N", help="Also time N predictions of the first image.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point: print the prediction for each image."""
    args = parse_args(argv)
    for path in args.images:
        result = predict_image(path, args.model_dir)
        if args.json:
            print(json.dumps({"image": path, **result}, indent=2))
            continue
        print(f"{path}: {result['label']} ({result['confidence']:.1%})   malignant probability {result['malignant_probability']:.1%}")
        for name, p in list(result["probabilities"].items())[:3]:
            print(f"    {name:<6} {p:.1%}")
    if args.benchmark:
        stats = benchmark(args.images[0], n_runs=args.benchmark, model_dir=args.model_dir)
        print(
            f"\nInference time: {stats['mean_ms']:.1f} ± {stats['std_ms']:.1f} ms per image "
            f"(median {stats['median_ms']:.1f} ms, {stats['n_runs']} runs, {stats['device']})"
            f"\nModel size on disk: {stats['model_size_mb']:.1f} MB"
        )


if __name__ == "__main__":
    main()
