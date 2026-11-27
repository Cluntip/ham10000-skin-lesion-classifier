"""Experiment tracking: one record per run, one experiment table for the team.

Every run (a neural model or a baseline, by any team member) is saved as its
own JSON file, ``experiments/runs/<run_id>.json``. Because every run has its
own file, two people training in different Colab sessions never edit the same
file, so their git pushes never conflict. :func:`build_log` combines all the
records into ``experiments/experiment_log.csv``, the experiment table required
by the spec (section 3.5): run ID, date, author, model, hyperparameters,
random seed, training time, parameters, training / validation metrics.

Typical use (``src/experiments.py`` and ``src/train.py`` already do this)::

    with Timer() as timer:
        model, history = train(config)
    log_run(config, model=model, history=history.history, train_time_s=timer.seconds)

Owner: Experimentation Engineer.
"""

from __future__ import annotations

import copy
import getpass
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, resolve_path
from src.curves import best_epoch_index, diagnose_fit, plot_learning_curves

#: Columns of ``experiments/experiment_log.csv``, in order. Metric columns
#: (``train_*`` / ``val_*``) found in the records are inserted before ``fit_diagnosis``.
LOG_COLUMNS = [
    "run_id", "date", "author", "study", "factor", "factor_value", "model", "seed",
    "optimizer", "learning_rate", "batch_size", "epochs_trained", "best_epoch",
    "train_time_s", "params_total", "params_trainable",
    "fit_diagnosis", "generalization_gap", "setup_hash", "config_file", "git_commit",
    "hyperparameters", "notes",
]

#: Metrics shown first in the log (others found in the records follow).
MAIN_METRICS = [
    "train_loss", "val_loss", "train_accuracy", "val_accuracy", "train_macro_f1", "val_macro_f1",
]


class Timer:
    """Context manager that measures wall-clock time in seconds.

    Example::

        with Timer() as timer:
            model.fit(...)
        print(timer.seconds)
    """

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.seconds = time.perf_counter() - self.start


def experiments_dir(config: dict) -> Path:
    """Return the absolute ``experiments/`` folder named in the config."""
    return resolve_path(config["paths"]["experiments_dir"])


def make_run_id(name: str, seed: int, now: datetime | None = None) -> str:
    """Build a unique, sortable run ID such as ``20261127-153012_lr_1e-4_s42``.

    Args:
        name: Experiment name (``experiment.name`` in the config).
        seed: Random seed of the run.
        now: Timestamp to use (default: current time).

    Returns:
        The run ID; it is also the file name of the run record.
    """
    now = now or datetime.now()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-") or "run"
    return f"{now:%Y%m%d-%H%M%S}_{slug}_s{seed}"


def flatten(nested: dict, prefix: str = "") -> dict:
    """Flatten a nested dictionary into dotted keys.

    Example: ``{"train": {"learning_rate": 0.001}}`` becomes
    ``{"train.learning_rate": 0.001}``.

    Args:
        nested: The dictionary to flatten.
        prefix: Prefix for the keys (used by the recursion).

    Returns:
        A flat dictionary.
    """
    flat = {}
    for key, value in nested.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def get_by_path(config: dict, dotted_key: str) -> Any:
    """Read a nested config value by its dotted key, e.g. ``"train.batch_size"``."""
    node = config
    for key in dotted_key.split("."):
        node = node[key]
    return node


def setup_hash(config: dict) -> str:
    """Fingerprint of a training setup that ignores the run's name and seed.

    Two runs with the same ``setup_hash`` used the same data, model and
    hyperparameters, so they differ only in their seed. This groups the
    3-seed runs of one configuration for the mean ± std table.

    Args:
        config: The full configuration of the run.

    Returns:
        A 10-character hexadecimal hash.
    """
    config = copy.deepcopy(config)
    for key in ("name", "seed", "study"):
        config.get("experiment", {}).pop(key, None)
    text = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def get_author(explicit: str | None = None) -> str:
    """Return who ran the experiment.

    Priority: the ``explicit`` argument, then the ``EXPERIMENT_AUTHOR``
    environment variable (set it once per Colab session with
    ``%env EXPERIMENT_AUTHOR=Amr``), then ``git config user.name``, then the
    operating-system user name.

    Args:
        explicit: Author name passed on the command line, if any.

    Returns:
        The author name.
    """
    if explicit:
        return explicit
    if os.environ.get("EXPERIMENT_AUTHOR"):
        return os.environ["EXPERIMENT_AUTHOR"]
    name = _git("config", "user.name")
    return name or getpass.getuser()


def get_git_commit() -> str:
    """Return the current short commit hash, with ``+dirty`` if tracked files changed.

    Returns:
        For example ``"70b2bfb"`` or ``"70b2bfb+dirty"``, or ``"unknown"``
        outside a git checkout.
    """
    commit = _git("rev-parse", "--short", "HEAD")
    if not commit:
        return "unknown"
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return f"{commit}+dirty" if dirty else commit


def _git(*args: str) -> str:
    """Run a git command in the repository and return its output ("" on failure)."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def count_params(model: Any) -> dict[str, int]:
    """Count the total, trainable and non-trainable parameters of a Keras model.

    Args:
        model: A built Keras model.

    Returns:
        ``{"total": ..., "trainable": ..., "non_trainable": ...}``; the total
        equals ``model.count_params()`` and the ``model.summary()`` total.
    """
    total = int(model.count_params())
    trainable = int(sum(np.prod(weight.shape) for weight in model.trainable_weights))
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


def summarise_history(history: dict) -> dict:
    """Extract the metrics of the best epoch from a Keras history.

    The best epoch is the one with the lowest validation loss, i.e. the
    weights that early stopping restores, so these are the numbers the model
    actually has after training. Training metrics get a ``train_`` prefix
    (``loss`` becomes ``train_loss``); validation metrics keep ``val_``.

    Args:
        history: A Keras history dictionary.

    Returns:
        ``epochs_trained``, ``best_epoch`` (1-based), the metrics at the best
        epoch and ``generalization_gap`` (validation loss minus training loss
        at the best epoch).
    """
    best = best_epoch_index(history)
    summary: dict[str, Any] = {"epochs_trained": len(history["loss"]), "best_epoch": best + 1}
    for key, values in history.items():
        if not values:
            continue
        name = key if key.startswith("val_") else f"train_{key}"
        summary[name] = float(values[best])
    if "val_loss" in summary and "train_loss" in summary:
        summary["generalization_gap"] = summary["val_loss"] - summary["train_loss"]
    return summary


def log_run(
    config: dict,
    *,
    history: dict | None = None,
    model: Any = None,
    params: dict | None = None,
    metrics: dict | None = None,
    train_time_s: float | None = None,
    study: str | None = None,
    factor: str | None = None,
    config_file: str | os.PathLike | None = None,
    author: str | None = None,
    notes: str = "",
    save_curves: bool = True,
    update_log: bool = True,
) -> dict:
    """Save one run as ``experiments/runs/<run_id>.json`` and refresh the log.

    Works for Keras runs (pass ``history`` and ``model``) and for baselines
    that have no training history (pass ``metrics`` such as
    ``{"val_macro_f1": 0.41}`` and, if known, ``params``).

    Args:
        config: The full configuration of the run.
        history: ``model.fit(...).history``, one value per epoch.
        model: The trained Keras model, used to count parameters.
        params: Parameter counts, if there is no Keras model (baselines).
        metrics: Extra or final metrics, e.g. from a baseline; they override
            the values taken from ``history``.
        train_time_s: Training time in seconds.
        study: Name of the tuning study / ablation this run belongs to.
        factor: Dotted config key the study varies, e.g. ``"train.learning_rate"``.
        config_file: The config file the run was started from.
        author: Who ran it (see :func:`get_author`).
        notes: Free-text remark.
        save_curves: Save the learning curves to ``experiments/curves/<run_id>.png``.
        update_log: Rebuild ``experiments/experiment_log.csv`` afterwards.

    Returns:
        The run record that was written.
    """
    seed = int(config["experiment"]["seed"])
    run_id = make_run_id(config["experiment"].get("name", "run"), seed)
    out_dir = experiments_dir(config)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    while (runs_dir / f"{run_id}.json").exists():  # two runs in the same second
        run_id += "b"

    history = {key: [float(v) for v in values] for key, values in (history or {}).items()}
    summary = summarise_history(history) if history.get("loss") else {}
    summary.update({key: float(value) for key, value in (metrics or {}).items()})
    if model is not None:
        params = count_params(model)

    record = {
        "run_id": run_id,
        "date": datetime.now().isoformat(timespec="seconds"),
        "author": get_author(author),
        "study": study or config["experiment"].get("study") or "",
        "factor": factor or "",
        "factor_value": get_by_path(config, factor) if factor else "",
        "model": config["model"]["name"],
        "seed": seed,
        "config_file": _relative(config_file) if config_file else "",
        "setup_hash": setup_hash(config),
        "git_commit": get_git_commit(),
        "train_time_s": round(float(train_time_s), 1) if train_time_s is not None else None,
        "params": params or {},
        "summary": summary,
        "fit_diagnosis": diagnose_fit(history),
        "notes": notes,
        "history": history,
        "config": config,
    }
    with open(runs_dir / f"{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    if save_curves and history.get("loss"):
        plot_learning_curves(history, title=run_id, save_path=out_dir / "curves" / f"{run_id}.png")
    if update_log:
        build_log(out_dir)
    return record


def _relative(path: str | os.PathLike) -> str:
    """Return ``path`` relative to the repository root when possible."""
    path = resolve_path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_runs(out_dir: str | os.PathLike) -> list[dict]:
    """Load every run record from ``<out_dir>/runs/``, oldest first.

    Args:
        out_dir: The experiments folder.

    Returns:
        A list of run records (dictionaries).
    """
    records = []
    for path in sorted(Path(out_dir, "runs").glob("*.json")):
        with open(path, encoding="utf-8") as f:
            records.append(json.load(f))
    return sorted(records, key=lambda r: (r["date"], r["run_id"]))


def record_to_row(record: dict) -> dict:
    """Turn one run record into one row of the experiment table."""
    config = record.get("config", {})
    train_cfg = config.get("train", {})
    hyper = {
        key: value
        for key, value in flatten(config).items()
        if key.split(".")[0] in ("model", "train") or key in ("data.image_size", "data.augmentation", "data.class_weights")
    }
    row = {
        **{key: record.get(key, "") for key in (
            "run_id", "date", "author", "study", "factor", "factor_value", "model", "seed",
            "train_time_s", "fit_diagnosis", "setup_hash", "config_file", "git_commit", "notes",
        )},
        "optimizer": train_cfg.get("optimizer", ""),
        "learning_rate": train_cfg.get("learning_rate", ""),
        "batch_size": train_cfg.get("batch_size", ""),
        "params_total": record.get("params", {}).get("total", ""),
        "params_trainable": record.get("params", {}).get("trainable", ""),
        "hyperparameters": json.dumps(hyper, sort_keys=True, default=str),
    }
    row.update(record.get("summary", {}))
    return row


def build_log(out_dir: str | os.PathLike, csv_name: str = "experiment_log.csv") -> pd.DataFrame:
    """Combine all run records into ``<out_dir>/experiment_log.csv``.

    The CSV is generated from the JSON records, so if git ever reports a
    conflict in it, simply run this function again (``python -m src.experiments log``).

    Args:
        out_dir: The experiments folder.
        csv_name: File name of the table.

    Returns:
        The experiment table as a DataFrame (one row per run).
    """
    rows = [record_to_row(record) for record in load_runs(out_dir)]
    table = pd.DataFrame(rows)
    metric_columns = [c for c in MAIN_METRICS if c in table.columns]
    metric_columns += sorted(
        c for c in table.columns
        if c.startswith(("train_", "val_")) and c not in metric_columns and c != "train_time_s"
    )
    split = LOG_COLUMNS.index("fit_diagnosis")
    columns = LOG_COLUMNS[:split] + metric_columns + LOG_COLUMNS[split:]
    table = table.reindex(columns=columns)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(out_dir) / csv_name, index=False)
    return table


def seed_summary(
    table: pd.DataFrame,
    metrics: Iterable[str] = ("val_macro_f1", "val_accuracy", "val_loss"),
    by: Iterable[str] = ("study", "factor_value", "setup_hash"),
) -> pd.DataFrame:
    """Mean ± standard deviation of each metric over the seeds of each setup.

    Runs are grouped by ``by`` (by default the study, the value of the varied
    factor and the setup hash, so only runs that differ in their seed are
    averaged). The standard deviation is the sample standard deviation
    (``ddof=1``); it is empty when a setup has a single seed.

    Args:
        table: The experiment table from :func:`build_log`.
        metrics: Metric columns to summarise (missing ones are skipped).
        by: Columns that identify one setup.

    Returns:
        One row per setup with ``n_seeds``, ``seeds``, ``<metric>_mean``,
        ``<metric>_std`` and a readable ``<metric>`` column such as
        ``"0.612 ± 0.013"``.
    """
    by = [c for c in by if c in table.columns]
    metrics = [m for m in metrics if m in table.columns]
    table = table.copy()
    for column in by:  # NaN never equals NaN, so empty keys would break the grouping
        table[column] = table[column].astype(object).where(table[column].notna(), "")

    rows = []
    for keys, group in table.groupby(by, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        row["model"] = group["model"].iloc[0]
        row["n_seeds"] = len(group)
        row["seeds"] = " ".join(str(s) for s in sorted(group["seed"]))
        for metric in metrics:
            values = group[metric].astype(float)
            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else float("nan")
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[metric] = f"{mean:.3f}" if np.isnan(std) else f"{mean:.3f} ± {std:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)
