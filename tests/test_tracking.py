"""Unit tests for src/tracking.py (no TensorFlow needed: a fake model is used)."""

import json
from datetime import datetime

import pandas as pd
import pytest

from src.config import load_config
from src.tracking import (
    LOG_COLUMNS,
    build_log,
    count_params,
    flatten,
    get_author,
    log_run,
    make_run_id,
    seed_summary,
    setup_hash,
    summarise_history,
)

HISTORY = {
    "loss": [1.2, 0.9, 0.7, 0.6],
    "accuracy": [0.5, 0.6, 0.7, 0.75],
    "macro_f1": [0.2, 0.3, 0.4, 0.45],
    "val_loss": [1.1, 0.8, 0.85, 0.9],
    "val_accuracy": [0.55, 0.65, 0.66, 0.64],
    "val_macro_f1": [0.25, 0.35, 0.36, 0.33],
}


class FakeWeight:
    def __init__(self, *shape):
        self.shape = shape


class FakeModel:
    """Stands in for a Keras model: 3x4 kernel + 4 bias trainable, 4 frozen."""

    trainable_weights = [FakeWeight(3, 4), FakeWeight(4)]

    def count_params(self):
        return 3 * 4 + 4 + 4


@pytest.fixture
def config(tmp_path):
    return load_config("configs/base.yaml", [f"paths.experiments_dir={tmp_path}"])


def test_run_id_is_sortable_and_safe():
    run_id = make_run_id("lr 1e-4/test", 7, now=datetime(2026, 11, 27, 15, 30, 12))
    assert run_id == "20261127-153012_lr-1e-4-test_s7"


def test_flatten_uses_dotted_keys():
    assert flatten({"train": {"lr": 0.1, "opt": {"name": "adam"}}}) == {"train.lr": 0.1, "train.opt.name": "adam"}


def test_setup_hash_ignores_name_and_seed_only(config):
    other_seed = load_config("configs/base.yaml", ["experiment.seed=7", "experiment.name=x"])
    other_seed["paths"] = config["paths"]
    other_lr = load_config("configs/base.yaml", ["train.learning_rate=0.01"])
    other_lr["paths"] = config["paths"]
    assert setup_hash(config) == setup_hash(other_seed)
    assert setup_hash(config) != setup_hash(other_lr)


def test_summary_uses_best_validation_epoch():
    summary = summarise_history(HISTORY)
    assert summary["best_epoch"] == 2 and summary["epochs_trained"] == 4
    assert summary["train_loss"] == 0.9 and summary["val_macro_f1"] == 0.35
    assert summary["generalization_gap"] == pytest.approx(0.8 - 0.9)


def test_count_params_matches_model():
    assert count_params(FakeModel()) == {"total": 20, "trainable": 16, "non_trainable": 4}


def test_author_priority(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_AUTHOR", "Amr")
    assert get_author() == "Amr"
    assert get_author("Ghada") == "Ghada"


def test_log_run_writes_record_curves_and_table(config, tmp_path):
    record = log_run(config, history=HISTORY, model=FakeModel(), train_time_s=12.34, study="lr",
                     factor="train.learning_rate", author="Amr")

    saved = json.loads((tmp_path / "runs" / f"{record['run_id']}.json").read_text())
    assert saved["author"] == "Amr" and saved["factor_value"] == 0.001
    assert saved["params"]["total"] == 20 and saved["train_time_s"] == 12.3
    assert saved["fit_diagnosis"] == "overfitting"  # val_loss rose 12.5 % after epoch 2
    assert (tmp_path / "curves" / f"{record['run_id']}.png").exists()

    table = pd.read_csv(tmp_path / "experiment_log.csv")
    assert len(table) == 1
    for column in ["run_id", "date", "author", "model", "seed", "train_time_s", "params_total",
                   "hyperparameters", "train_loss", "val_loss", "val_macro_f1"]:
        assert column in table.columns
    assert json.loads(table.loc[0, "hyperparameters"])["train.optimizer"] == "adam"


def test_log_run_for_a_baseline_without_history(config, tmp_path):
    config["model"]["name"] = "majority"
    log_run(config, metrics={"val_macro_f1": 0.115, "val_accuracy": 0.670}, params={"total": 0}, author="Ammar")
    table = build_log(tmp_path)
    assert table.loc[0, "model"] == "majority"
    assert table.loc[0, "val_macro_f1"] == pytest.approx(0.115)
    assert table.loc[0, "fit_diagnosis"] == "n/a"


def test_log_columns_come_first_in_order(config, tmp_path):
    log_run(config, history=HISTORY, model=FakeModel(), save_curves=False)
    table = build_log(tmp_path)
    fixed = [c for c in table.columns if c in LOG_COLUMNS]
    assert fixed == LOG_COLUMNS


def test_seed_summary_mean_and_std():
    table = pd.DataFrame({
        "study": ["final"] * 3, "factor_value": [""] * 3, "setup_hash": ["abc"] * 3,
        "model": ["resnet50_finetune"] * 3, "seed": [44, 42, 43], "val_macro_f1": [0.60, 0.62, 0.64],
    })
    summary = seed_summary(table, metrics=["val_macro_f1"])
    assert summary.loc[0, "n_seeds"] == 3 and summary.loc[0, "seeds"] == "42 43 44"
    assert summary.loc[0, "val_macro_f1_mean"] == pytest.approx(0.62)
    assert summary.loc[0, "val_macro_f1_std"] == pytest.approx(0.02)
    assert summary.loc[0, "val_macro_f1"] == "0.620 ± 0.020"
