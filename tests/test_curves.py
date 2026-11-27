"""Unit tests for src/curves.py."""

from src.curves import best_epoch_index, diagnose_fit, plot_learning_curves, plot_study

OVERFIT = {"loss": [1.0, 0.7, 0.5, 0.3, 0.2], "val_loss": [1.1, 0.8, 0.9, 1.0, 1.2]}
STILL_IMPROVING = {"loss": [1.0, 0.8, 0.6], "val_loss": [1.1, 0.9, 0.7]}
GOOD_FIT = {"loss": [1.0, 0.7, 0.6, 0.58], "val_loss": [1.0, 0.75, 0.70, 0.72]}


def test_best_epoch_is_lowest_val_loss():
    assert best_epoch_index(OVERFIT) == 1
    assert best_epoch_index({"loss": [3, 2, 1]}) == 2  # no validation: last epoch


def test_diagnose_fit_labels():
    assert diagnose_fit(OVERFIT) == "overfitting"
    assert diagnose_fit(STILL_IMPROVING) == "underfitting / not converged"
    assert diagnose_fit(GOOD_FIT) == "good fit"
    assert diagnose_fit({"loss": [1.0, 0.5]}) == "n/a"


def test_val_loss_rise_without_falling_train_loss_is_not_overfitting():
    noisy = {"loss": [1.0, 0.6, 0.65, 0.7], "val_loss": [1.0, 0.6, 0.7, 0.8]}
    assert diagnose_fit(noisy) == "good fit"


def test_plot_learning_curves_saves_png(tmp_path):
    history = {**OVERFIT, "macro_f1": [0.1, 0.2, 0.3, 0.4, 0.5], "val_macro_f1": [0.1, 0.2, 0.2, 0.2, 0.1]}
    path = tmp_path / "curves" / "run.png"
    plot_learning_curves(history, title="run", save_path=path)
    assert path.exists() and path.stat().st_size > 0


def test_plot_study_saves_png(tmp_path):
    runs = [
        {"run_id": "a", "seed": 42, "factor": "train.learning_rate", "factor_value": 0.01, "history": OVERFIT},
        {"run_id": "b", "seed": 42, "factor": "train.learning_rate", "factor_value": 0.001, "history": GOOD_FIT},
    ]
    path = tmp_path / "study.png"
    plot_study(runs, metric="val_loss", save_path=path)
    assert path.exists()
