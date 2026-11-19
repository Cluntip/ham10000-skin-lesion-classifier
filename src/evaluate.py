"""Test-set evaluation and error analysis.

Computes accuracy, the confusion matrix, per-class precision / recall / F1,
macro-F1 and one-vs-rest ROC curves with AUC, for the final model and the
baselines. This is the only module that reads the test split, and it is run
once, at the end of the project.

Owner: Evaluation & Deployment Engineer. The functions below are the agreed
interface; their bodies are to be implemented.
"""

from __future__ import annotations

import numpy as np


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict:
    """Compute the full classification metric set.

    Args:
        y_true: Integer class labels, shape ``(n,)``.
        y_prob: Predicted class probabilities, shape ``(n, n_classes)``.
        class_names: Class names in label order.

    Returns:
        A dictionary of metrics (overall and per class).
    """
    raise NotImplementedError


def find_worst_errors(y_true: np.ndarray, y_prob: np.ndarray, k: int = 20) -> np.ndarray:
    """Return the indices of the ``k`` most confident misclassifications.

    Args:
        y_true: Integer class labels, shape ``(n,)``.
        y_prob: Predicted class probabilities, shape ``(n, n_classes)``.
        k: Number of samples to return.

    Returns:
        Indices into the evaluated split.
    """
    raise NotImplementedError


def main() -> None:
    """Command-line entry point: evaluate the final model on the test split."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
