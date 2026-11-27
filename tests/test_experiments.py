"""Unit tests for src/experiments.py (training is replaced by a fake function)."""

import pandas as pd
import pytest

from src.config import CONFIG_DIR, load_config
from src.experiments import check_one_factor, load_studies, main, run_one
from src.tracking import get_by_path

FAKE_HISTORY = {
    "loss": [1.0, 0.8, 0.7],
    "val_loss": [1.0, 0.9, 0.95],
    "macro_f1": [0.2, 0.3, 0.35],
    "val_macro_f1": [0.2, 0.28, 0.27],
}


class FakeModel:
    trainable_weights = []

    def count_params(self):
        return 0


def fake_train(config):
    """Pretend to train: the score depends on the seed so seeds differ."""
    history = {**FAKE_HISTORY, "val_macro_f1": [v + config["experiment"]["seed"] / 1000 for v in FAKE_HISTORY["val_macro_f1"]]}
    return FakeModel(), history


@pytest.fixture
def config_file(tmp_path):
    """A config that extends base.yaml but logs into a temporary folder."""
    path = tmp_path / "tmp_config.yaml"
    path.write_text(f"extends: {CONFIG_DIR / 'base.yaml'}\npaths:\n  experiments_dir: {tmp_path / 'experiments'}\n")
    return path


def test_every_study_changes_exactly_one_factor():
    studies = load_studies()
    assert {"learning_rate", "batch_size", "optimizer"} <= set(studies)
    for name, study in studies.items():
        check_one_factor(name, study)  # raises if a config changes anything else
        assert len(study["configs"]) >= 3
        assert "base.yaml" in study["configs"]  # every study includes the reference


def test_study_values():
    studies = load_studies()
    values = {
        name: sorted(str(get_by_path(load_config(CONFIG_DIR / f), s["factor"])) for f in s["configs"])
        for name, s in studies.items()
    }
    assert values["optimizer"] == ["adam", "sgd", "sgd_momentum"]
    assert values["learning_rate"] == ["0.0001", "0.001", "0.01"]
    assert values["batch_size"] == ["16", "32", "64"]


def test_check_one_factor_rejects_a_second_change(tmp_path):
    (tmp_path / "ref.yaml").write_text("experiment: {name: ref, seed: 1}\ntrain: {learning_rate: 0.1, batch_size: 32}\n")
    (tmp_path / "bad.yaml").write_text("extends: ref.yaml\nexperiment: {name: bad}\ntrain: {learning_rate: 0.01, batch_size: 64}\n")
    study = {"factor": "train.learning_rate", "configs": ["ref.yaml", "bad.yaml"]}
    with pytest.raises(ValueError, match="batch_size"):
        check_one_factor("lr", study, studies_file=tmp_path / "studies.yaml")


def test_dry_run_lists_the_plan(capsys):
    main(["run", "--study", "learning_rate", "--seeds", "1", "2", "--dry-run"])
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[")]
    assert len(lines) == 6  # 3 configs x 2 seeds
    assert "lr_1e-2.yaml" in lines[0]


def test_run_one_logs_and_skip_existing(config_file, tmp_path):
    record = run_one(config_file, seed=42, study="demo", author="Amr", train_fn=fake_train)
    assert record["study"] == "demo" and record["seed"] == 42
    assert run_one(config_file, seed=42, study="demo", skip_existing=True, train_fn=fake_train) is None
    assert run_one(config_file, seed=43, study="demo", skip_existing=True, train_fn=fake_train) is not None
    table = pd.read_csv(tmp_path / "experiments" / "experiment_log.csv")
    assert sorted(table["seed"]) == [42, 43]


def test_log_plot_and_summary_commands(config_file, tmp_path, capsys):
    for seed in (42, 43, 44):
        run_one(config_file, seed=seed, study="final", author="Ammar", train_fn=fake_train)

    main(["log", "--config", str(config_file)])
    main(["plot", "--config", str(config_file), "--study", "final", "--metric", "val_macro_f1"])
    main(["summary", "--config", str(config_file), "--study", "final"])

    out_dir = tmp_path / "experiments"
    assert (out_dir / "curves" / "study_final_val_macro_f1.png").exists()
    summary = pd.read_csv(out_dir / "summary_final.csv")
    assert summary.loc[0, "n_seeds"] == 3
    assert "±" in summary.loc[0, "val_macro_f1"]
    assert "±" in capsys.readouterr().out
