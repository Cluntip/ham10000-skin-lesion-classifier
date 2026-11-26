"""Unit tests for src/evaluate.py (no TensorFlow needed)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    build_error_table,
    compute_metrics,
    error_rate_by,
    evaluate_predictions,
    find_worst_errors,
    malignant_recall,
    most_common_confusions,
)

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def one_hot_probs(labels, n_classes=7, confidence=0.9):
    """Probabilities that put `confidence` on the given labels."""
    probs = np.full((len(labels), n_classes), (1 - confidence) / (n_classes - 1))
    probs[np.arange(len(labels)), labels] = confidence
    return probs


def test_perfect_predictions_score_one():
    y = np.repeat(np.arange(7), 3)
    m = compute_metrics(y, one_hot_probs(y), CLASSES)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["macro_auc_ovr"] == 1.0
    assert m["melanoma_recall"] == 1.0
    assert m["malignant_recall"] == 1.0
    assert np.array_equal(np.diag(m["confusion_matrix"]), [3] * 7)
    json.dumps(m)  # must be JSON-serialisable


def test_majority_baseline_has_high_accuracy_but_low_macro_f1():
    # 90 nv + 10 mel; always predicting nv is the trivial baseline.
    y = np.array([5] * 90 + [4] * 10)
    m = compute_metrics(y, one_hot_probs(np.full(100, 5)), CLASSES)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["melanoma_recall"] == 0.0
    assert m["macro_f1"] < 0.5


def test_malignant_recall_counts_malignant_confusions_as_referrals():
    y_true = np.array([4, 4, 1, 5])  # mel, mel, bcc, nv
    y_pred = np.array([1, 5, 1, 5])  # mel->bcc (still referred), mel->nv (missed)
    assert malignant_recall(y_true, y_pred, CLASSES) == pytest.approx(2 / 3)


def test_wrong_shapes_raise():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1]), np.zeros((3, 7)), CLASSES)
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1]), np.zeros((2, 5)), CLASSES)


def test_find_worst_errors_orders_by_confidence():
    y_true = np.array([0, 1, 2, 3])
    y_prob = np.array(
        [
            [0.9, 0.1, 0, 0, 0, 0, 0],    # correct
            [0.6, 0.4, 0, 0, 0, 0, 0],    # wrong, confidence 0.6
            [0.95, 0, 0.05, 0, 0, 0, 0],  # wrong, confidence 0.95
            [0, 0, 0, 1.0, 0, 0, 0],      # correct
        ]
    )
    assert find_worst_errors(y_true, y_prob, k=5).tolist() == [2, 1]
    assert find_worst_errors(y_true, y_prob, k=1).tolist() == [2]


def test_most_common_confusions():
    cm = np.zeros((7, 7), dtype=int)
    cm[4, 5] = 8   # mel -> nv
    cm[2, 4] = 3   # bkl -> mel
    table = most_common_confusions(cm, CLASSES, top=2)
    assert table.iloc[0][["true", "predicted", "count"]].tolist() == ["mel", "nv", 8]
    assert len(table) == 2


def test_error_table_and_rate_by_use_metadata():
    y_true = np.array([4, 5, 4])
    y_prob = one_hot_probs(np.array([5, 5, 4]))
    meta = pd.DataFrame({"image_id": ["a", "b", "c"], "dx_type": ["histo", "consensus", "histo"]})
    table = build_error_table(find_worst_errors(y_true, y_prob), y_true, y_prob, CLASSES, meta)
    assert table.loc[0, "image_id"] == "a"
    assert table.loc[0, "true"] == "mel" and table.loc[0, "predicted"] == "nv"
    assert "failure_category" in table

    rates = error_rate_by(meta, y_true, y_prob.argmax(1), "dx_type")
    assert rates.set_index("dx_type").loc["histo", "error_rate"] == pytest.approx(0.5)


def test_evaluate_predictions_writes_all_files(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, size=60)
    probs = rng.dirichlet(np.ones(7), size=60)
    evaluate_predictions(y, probs, CLASSES, tmp_path, name="random")
    for name in ("metrics.json", "confusion_matrix.png", "roc_curves.png", "top_confusions.csv", "worst_errors.csv"):
        assert (tmp_path / name).exists(), name
