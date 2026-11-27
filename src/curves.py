"""Learning-curve plots and a simple over/underfitting diagnosis.

A *history* here is the dictionary Keras returns in ``model.fit(...).history``
(and that :mod:`src.tracking` stores in every run record), for example
``{"loss": [...], "val_loss": [...], "macro_f1": [...], "val_macro_f1": [...]}``
with one value per epoch.

Owner: Experimentation Engineer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

#: Relative rise of the validation loss above its minimum that counts as overfitting.
OVERFIT_RISE = 0.10


def best_epoch_index(history: dict) -> int:
    """Return the 0-based index of the epoch with the lowest validation loss.

    This is the epoch whose weights early stopping restores
    (``restore_best_weights=True``). Without ``val_loss`` the last epoch is used.

    Args:
        history: A Keras history dictionary.

    Returns:
        The index of the best epoch.
    """
    if history.get("val_loss"):
        return int(np.argmin(history["val_loss"]))
    return len(history["loss"]) - 1


def diagnose_fit(history: dict, overfit_rise: float = OVERFIT_RISE) -> str:
    """Label a training run as overfitting, underfitting or a good fit.

    The rules, applied to the loss curves:

    1. **underfitting / not converged**: the validation loss is still at its
       lowest on the last epoch, so the model was still improving when
       training stopped (train longer, raise the learning rate or add capacity).
    2. **overfitting**: after the best epoch the validation loss rose by more
       than ``overfit_rise`` (10 % by default) while the training loss kept
       falling, so the model memorises the training set.
    3. **good fit**: otherwise. The validation loss levelled off and early
       stopping ended the run near its minimum.

    This is a first label for the experiment table; the curves are still
    inspected and discussed in the report.

    Args:
        history: A Keras history dictionary with ``loss`` and ``val_loss``.
        overfit_rise: Relative rise of ``val_loss`` above its minimum that
            counts as overfitting.

    Returns:
        One of ``"overfitting"``, ``"underfitting / not converged"``,
        ``"good fit"``, or ``"n/a"`` when there is no validation loss.
    """
    val_loss = history.get("val_loss")
    train_loss = history.get("loss")
    if not val_loss or not train_loss:
        return "n/a"

    best = best_epoch_index(history)
    if best == len(val_loss) - 1:
        return "underfitting / not converged"

    rise = (val_loss[-1] - val_loss[best]) / abs(val_loss[best])
    train_kept_falling = train_loss[-1] < train_loss[best]
    if rise > overfit_rise and train_kept_falling:
        return "overfitting"
    return "good fit"


def plot_learning_curves(
    history: dict,
    title: str = "",
    metrics: Sequence[str] = ("loss", "macro_f1"),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training vs validation curves for one run, one panel per metric.

    A dashed vertical line marks the best epoch (lowest validation loss), and
    the fit diagnosis is added to the title.

    Args:
        history: A Keras history dictionary.
        title: Figure title, e.g. the run ID.
        metrics: Metric names (without the ``val_`` prefix) to plot; metrics
            missing from the history are skipped.
        save_path: If given, the figure is saved there (PNG) and closed.

    Returns:
        The Matplotlib figure.
    """
    metrics = [m for m in metrics if m in history]
    fig, axes = plt.subplots(1, max(len(metrics), 1), figsize=(5.5 * max(len(metrics), 1), 4), squeeze=False)
    epochs = np.arange(1, len(history["loss"]) + 1)
    best = best_epoch_index(history) + 1

    for ax, metric in zip(axes[0], metrics):
        ax.plot(epochs, history[metric], label="train", color="tab:blue")
        if f"val_{metric}" in history:
            ax.plot(epochs, history[f"val_{metric}"], label="validation", color="tab:orange")
        ax.axvline(best, color="gray", linestyle="--", linewidth=1, label=f"best epoch ({best})")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(f"{title}  [{diagnose_fit(history)}]".strip())
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_study(
    runs: Sequence[dict],
    metric: str = "val_loss",
    title: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Overlay one metric for several runs, e.g. the 3 learning rates of a study.

    Args:
        runs: Run records from :func:`src.tracking.load_runs`; each needs a
            ``history`` and is labelled by its ``factor_value`` (or run ID).
        metric: History key to plot, e.g. ``"val_loss"`` or ``"val_macro_f1"``.
        title: Figure title.
        save_path: If given, the figure is saved there (PNG) and closed.

    Returns:
        The Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run in runs:
        values = run["history"].get(metric)
        if not values:
            continue
        label = f"{run.get('factor')}={run.get('factor_value')}" if run.get("factor") else run["run_id"]
        label += f" (seed {run['seed']})"
        ax.plot(np.arange(1, len(values) + 1), values, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def _save_or_return(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    """Save ``fig`` to ``save_path`` (creating folders) and close it, if a path is given."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    return fig
