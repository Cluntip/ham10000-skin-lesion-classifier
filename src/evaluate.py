"""Evaluation and error analysis.

Computes accuracy, the confusion matrix, per-class precision / recall / F1,
macro-F1 and one-vs-rest ROC curves with AUC, plus the clinically important
recalls (melanoma, and malignant-vs-benign), and collects the most confident
mistakes for the error analysis.

Two ways to use it:

* From Python, with predictions you already have (baselines, notebooks)::

      from src.evaluate import evaluate_predictions
      evaluate_predictions(y_true, y_prob, class_names, out_dir, metadata=df)

* From the command line, with a saved Keras model::

      python -m src.evaluate --config configs/base.yaml --model path/to/model.keras            # validation split
      python -m src.evaluate --config configs/final.yaml --split test                         # final model, test split

The **validation** split is the default. The test split must be asked for
explicitly with ``--split test`` and is used once, at the end, for the final
model and the baselines (spec section 3.6 and the test-set discipline rule).

This module imports TensorFlow and ``src.data`` only inside :func:`main`, so
the metric functions can be unit-tested without TensorFlow.

Owner: Evaluation & Deployment Engineer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

#: Classes that are malignant or pre-malignant in HAM10000. Missing one of
#: these (predicting a benign class instead) is the most expensive error.
MALIGNANT_CLASSES: tuple[str, ...] = ("akiec", "bcc", "mel")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _check_inputs(y_true: np.ndarray, y_prob: np.ndarray, class_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Validate shapes and return ``y_true`` as int and ``y_prob`` as float arrays."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim != 2 or y_prob.shape[0] != y_true.shape[0]:
        raise ValueError(f"y_prob must have shape (n, n_classes); got {y_prob.shape} for {y_true.shape[0]} labels.")
    if y_prob.shape[1] != len(class_names):
        raise ValueError(f"y_prob has {y_prob.shape[1]} columns but there are {len(class_names)} class names.")
    return y_true, y_prob


def malignant_recall(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]) -> float:
    """Share of malignant lesions that the model sends to *any* malignant class.

    This is the screening question "did we flag it for a biopsy?": confusing
    melanoma with basal cell carcinoma is still a correct referral, while
    calling it a nevus is a miss.

    Args:
        y_true: Integer class labels, shape ``(n,)``.
        y_pred: Integer predicted labels, shape ``(n,)``.
        class_names: Class names in label order.

    Returns:
        The recall in ``[0, 1]``, or ``nan`` if there are no malignant samples.
    """
    malignant_ids = [i for i, c in enumerate(class_names) if c in MALIGNANT_CLASSES]
    is_malignant = np.isin(y_true, malignant_ids)
    if not is_malignant.any():
        return float("nan")
    return float(np.isin(y_pred[is_malignant], malignant_ids).mean())


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict:
    """Compute the full classification metric set.

    Args:
        y_true: Integer class labels, shape ``(n,)``.
        y_prob: Predicted class probabilities, shape ``(n, n_classes)``.
        class_names: Class names in label order.

    Returns:
        A JSON-serialisable dictionary with the overall metrics
        (``accuracy``, ``balanced_accuracy``, ``macro_f1``, ``weighted_f1``,
        ``macro_auc_ovr``, ``melanoma_recall``, ``malignant_recall``), a
        ``per_class`` table (precision, recall, F1, AUC, support) and the raw
        ``confusion_matrix`` (rows = true class, columns = predicted class).
    """
    y_true, y_prob = _check_inputs(y_true, y_prob, class_names)
    n_classes = len(class_names)
    labels = list(range(n_classes))
    y_pred = y_prob.argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class_auc = []
    for i in labels:
        positives = y_true == i
        # AUC is undefined when a class is absent (or is the only class).
        if positives.all() or not positives.any():
            per_class_auc.append(float("nan"))
        else:
            per_class_auc.append(float(roc_auc_score(positives, y_prob[:, i])))
    present = [a for a in per_class_auc if not np.isnan(a)]

    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "auc": per_class_auc[i],
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }

    return {
        "n_samples": int(y_true.shape[0]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        # Mean of the one-vs-rest AUCs over the classes present in y_true.
        "macro_auc_ovr": float(np.mean(present)) if present else float("nan"),
        "melanoma_recall": per_class["mel"]["recall"] if "mel" in per_class else float("nan"),
        "malignant_recall": malignant_recall(y_true, y_pred, class_names),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": list(class_names),
    }


def most_common_confusions(cm: np.ndarray, class_names: Sequence[str], top: int = 5) -> pd.DataFrame:
    """List the largest off-diagonal cells of a confusion matrix.

    Args:
        cm: Confusion matrix, rows = true class, columns = predicted class.
        class_names: Class names in label order.
        top: How many confusions to return.

    Returns:
        A table with columns ``true``, ``predicted``, ``count`` and
        ``share_of_true`` (count divided by the number of samples of the true
        class), sorted by ``count`` descending.
    """
    cm = np.asarray(cm)
    rows = []
    for i, true_name in enumerate(class_names):
        row_total = cm[i].sum()
        for j, pred_name in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                rows.append(
                    {
                        "true": true_name,
                        "predicted": pred_name,
                        "count": int(cm[i, j]),
                        "share_of_true": float(cm[i, j] / row_total) if row_total else 0.0,
                    }
                )
    table = pd.DataFrame(rows, columns=["true", "predicted", "count", "share_of_true"])
    return table.sort_values("count", ascending=False).head(top).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------


def find_worst_errors(y_true: np.ndarray, y_prob: np.ndarray, k: int = 20) -> np.ndarray:
    """Return the indices of the ``k`` most confident misclassifications.

    "Most confident" means the highest predicted probability for the (wrong)
    predicted class: these are the mistakes the model is surest about, and
    usually the most informative ones (look-alike classes or label noise).

    Args:
        y_true: Integer class labels, shape ``(n,)``.
        y_prob: Predicted class probabilities, shape ``(n, n_classes)``.
        k: Number of samples to return.

    Returns:
        Indices into the evaluated split, most confident first. Fewer than
        ``k`` are returned if there are fewer mistakes.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = y_prob.argmax(axis=1)
    wrong = np.flatnonzero(y_pred != y_true)
    confidence = y_prob[wrong, y_pred[wrong]]
    # Stable sort so ties keep dataset order (reproducible tables).
    order = np.argsort(-confidence, kind="stable")
    return wrong[order[:k]]


def build_error_table(
    indices: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Describe selected samples for the error analysis.

    Args:
        indices: Sample indices, e.g. from :func:`find_worst_errors`.
        y_true: Integer class labels, shape ``(n,)``.
        y_prob: Predicted class probabilities, shape ``(n, n_classes)``.
        class_names: Class names in label order.
        metadata: Optional per-sample metadata in the **same row order** as
            ``y_true`` (e.g. the evaluated split from ``src.data``). Useful
            columns such as ``image_id``, ``lesion_id``, ``image_path``,
            ``dx_type``, ``age``, ``sex`` and ``localization`` are copied.

    Returns:
        One row per sample with the true and predicted class, the confidence,
        the probability given to the true class, the top-3 classes, and an
        empty ``failure_category`` / ``explanation`` column to fill in by
        hand while looking at the images.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float)
    names = np.asarray(class_names)
    rows = []
    for idx in indices:
        probs = y_prob[idx]
        pred = int(probs.argmax())
        top3 = np.argsort(-probs)[:3]
        rows.append(
            {
                "index": int(idx),
                "true": names[y_true[idx]],
                "predicted": names[pred],
                "confidence": float(probs[pred]),
                "p_true": float(probs[y_true[idx]]),
                "top3": ", ".join(f"{names[c]}={probs[c]:.2f}" for c in top3),
            }
        )
    table = pd.DataFrame(rows, columns=["index", "true", "predicted", "confidence", "p_true", "top3"])

    if metadata is not None and len(table):
        keep = [c for c in ("image_id", "lesion_id", "image_path", "dx_type", "age", "sex", "localization") if c in metadata]
        extra = metadata.iloc[table["index"].to_numpy()][keep].reset_index(drop=True)
        table = pd.concat([table, extra], axis=1)

    table["failure_category"] = ""
    table["explanation"] = ""
    return table


def error_rate_by(metadata: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, column: str) -> pd.DataFrame:
    """Error rate grouped by one metadata column (e.g. ``dx_type`` or ``localization``).

    Helps explain *why* the model fails: for example, if lesions confirmed
    only by "consensus" are misclassified more often than those confirmed by
    histopathology, label noise is a likely cause.

    Args:
        metadata: Per-sample metadata in the same row order as ``y_true``.
        y_true: Integer class labels.
        y_pred: Integer predicted labels.
        column: The metadata column to group by.

    Returns:
        A table with ``n`` and ``error_rate`` per group, highest error first.
    """
    wrong = np.asarray(y_pred) != np.asarray(y_true)
    frame = pd.DataFrame({column: metadata[column].to_numpy(), "wrong": wrong})
    table = frame.groupby(column, dropna=False)["wrong"].agg(n="size", error_rate="mean")
    return table.sort_values("error_rate", ascending=False).reset_index()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _pyplot():
    """Import Matplotlib with a non-interactive backend (works on servers and in tests)."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], path: str | Path, title: str = "") -> Path:
    """Save the confusion matrix, raw counts and row-normalised (recall) side by side.

    Args:
        cm: Confusion matrix, rows = true class, columns = predicted class.
        class_names: Class names in label order.
        path: Output PNG path.
        title: Figure title.

    Returns:
        The path of the saved figure.
    """
    plt = _pyplot()
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalised = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, matrix, fmt, subtitle in (
        (axes[0], cm, "d", "Counts"),
        (axes[1], normalised, ".2f", "Normalised by true class (diagonal = recall)"),
    ):
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(class_names)), class_names, rotation=45)
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(subtitle)
        threshold = matrix.max() / 2 if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j, i, format(matrix[i, j], fmt), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black", fontsize=8,
                )
        fig.colorbar(im, ax=ax, fraction=0.046)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, class_names: Sequence[str], path: str | Path, title: str = "") -> Path:
    """Save one-vs-rest ROC curves (one per class) with their AUC in the legend.

    Args:
        y_true: Integer class labels, shape ``(n,)``.
        y_prob: Predicted class probabilities, shape ``(n, n_classes)``.
        class_names: Class names in label order.
        path: Output PNG path.
        title: Figure title.

    Returns:
        The path of the saved figure.
    """
    plt = _pyplot()
    y_true, y_prob = _check_inputs(y_true, y_prob, class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        positives = y_true == i
        if positives.all() or not positives.any():
            continue
        fpr, tpr, _ = roc_curve(positives, y_prob[:, i])
        auc = roc_auc_score(positives, y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(title or "One-vs-rest ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_error_gallery(error_table: pd.DataFrame, path: str | Path, n_cols: int = 5) -> Path | None:
    """Save a grid of the misclassified images with their true / predicted class.

    Needs an ``image_path`` column in ``error_table`` (pass ``metadata`` to
    :func:`build_error_table`).

    Args:
        error_table: Output of :func:`build_error_table`.
        path: Output PNG path.
        n_cols: Images per row.

    Returns:
        The path of the saved figure, or ``None`` if there is nothing to draw.
    """
    if error_table.empty or "image_path" not in error_table:
        return None
    from PIL import Image

    plt = _pyplot()
    n_rows = int(np.ceil(len(error_table) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.3 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row in zip(axes.ravel(), error_table.itertuples()):
        with Image.open(row.image_path) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"#{row.index}  true={row.true}\npred={row.predicted} ({row.confidence:.2f})", fontsize=9)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# One call that writes everything for one model
# ---------------------------------------------------------------------------


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    out_dir: str | Path,
    name: str = "model",
    metadata: pd.DataFrame | None = None,
    k_errors: int = 20,
) -> dict:
    """Evaluate one model's predictions and write every artefact to ``out_dir``.

    Writes ``metrics.json``, ``confusion_matrix.png``, ``roc_curves.png``,
    ``top_confusions.csv``, ``worst_errors.csv`` and (if image paths are in
    ``metadata``) ``worst_errors.png``. Works for the neural models and the
    baselines alike, because it only needs probabilities.

    Args:
        y_true: Integer class labels, shape ``(n,)``.
        y_prob: Predicted class probabilities, shape ``(n, n_classes)``.
        class_names: Class names in label order.
        out_dir: Folder for the outputs (created if needed).
        name: Model name used in titles.
        metadata: Optional per-sample metadata, same row order as ``y_true``.
        k_errors: How many confident mistakes to collect.

    Returns:
        The metrics dictionary from :func:`compute_metrics`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(y_true, y_prob, class_names)
    metrics["model"] = name
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = np.asarray(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png", title=name)
    plot_roc_curves(y_true, y_prob, class_names, out_dir / "roc_curves.png", title=f"{name}: one-vs-rest ROC")
    most_common_confusions(cm, class_names, top=10).to_csv(out_dir / "top_confusions.csv", index=False)

    errors = build_error_table(find_worst_errors(y_true, y_prob, k_errors), y_true, y_prob, class_names, metadata)
    errors.to_csv(out_dir / "worst_errors.csv", index=False)
    plot_error_gallery(errors, out_dir / "worst_errors.png")
    return metrics


def format_summary(metrics: dict) -> str:
    """One-screen text summary of a metrics dictionary (for the console and logs)."""
    lines = [
        f"Model: {metrics.get('model', '?')}   samples: {metrics['n_samples']}",
        f"  macro-F1          {metrics['macro_f1']:.4f}   (primary metric)",
        f"  melanoma recall   {metrics['melanoma_recall']:.4f}",
        f"  malignant recall  {metrics['malignant_recall']:.4f}",
        f"  accuracy          {metrics['accuracy']:.4f}",
        f"  balanced accuracy {metrics['balanced_accuracy']:.4f}",
        f"  macro AUC (OvR)   {metrics['macro_auc_ovr']:.4f}",
        "",
        f"  {'class':<7}{'prec':>7}{'recall':>8}{'f1':>7}{'auc':>7}{'n':>6}",
    ]
    for name, m in metrics["per_class"].items():
        lines.append(f"  {name:<7}{m['precision']:>7.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}{m['auc']:>7.3f}{m['support']:>6d}")
    return "\n".join(lines)


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
    parser = argparse.ArgumentParser(description="Evaluate a saved model on the validation or test split.")
    parser.add_argument("--config", required=True, help="Path to a YAML file in configs/.")
    parser.add_argument("--model", default=None, help="Saved .keras model (default: models/final_model.keras).")
    parser.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="Split to evaluate. 'test' is for the FINAL model and the baselines only, once.",
    )
    parser.add_argument("--name", default=None, help="Name for titles and the output folder (default: model file name).")
    parser.add_argument("--k-errors", type=int, default=20, help="Number of confident mistakes to collect.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Config overrides.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point: evaluate a saved model and write the report files.

    Outputs go to ``<experiments_dir>/eval/<name>_<split>/``.
    """
    import keras

    from src import data
    from src.config import load_config, resolve_path

    args = parse_args(argv)
    config = load_config(args.config, args.set)
    class_names = list(config["data"]["classes"])

    model_path = resolve_path(args.model) if args.model else resolve_path(config["paths"]["models_dir"]) / "final_model.keras"
    name = args.name or model_path.stem
    model = keras.models.load_model(model_path)

    # The tf.data split must NOT be shuffled, so that its order matches the
    # metadata rows returned by split_by_lesion (agreed with the Data Engineer).
    dataset = data.make_datasets(config)[args.split]
    metadata = data.split_by_lesion(data.load_metadata(config), config)[args.split].reset_index(drop=True)

    y_prob = model.predict(dataset, verbose=1)
    y_true = np.concatenate([np.argmax(labels, axis=1) for _, labels in dataset])
    if len(y_true) != len(metadata):
        raise RuntimeError(f"Dataset has {len(y_true)} samples but metadata has {len(metadata)} rows.")
    if not np.array_equal(y_true, metadata["dx"].map({c: i for i, c in enumerate(class_names)}).to_numpy()):
        raise RuntimeError("Dataset order does not match the metadata order; is the split shuffled?")

    out_dir = resolve_path(config["paths"]["experiments_dir"]) / "eval" / f"{name}_{args.split}"
    metrics = evaluate_predictions(y_true, y_prob, class_names, out_dir, name=name, metadata=metadata, k_errors=args.k_errors)
    print(format_summary(metrics))
    print(f"\nWrote evaluation files to {out_dir}")


if __name__ == "__main__":
    main()
