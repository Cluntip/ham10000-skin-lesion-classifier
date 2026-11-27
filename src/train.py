"""Training entry point for the neural models.

Usage::

    python -m src.train --config configs/base.yaml
    python -m src.train --config configs/base.yaml --set train.batch_size=64 --seed 7

The script only ever touches the train and validation splits. The test split
is reserved for ``src/evaluate.py`` and is used once, at the end.

Owner: Tech Lead (entry point). Experiment logging and learning-curve plots
are added by the Experimentation Engineer.
"""

from __future__ import annotations

import argparse
from typing import Sequence

import keras

from src import data, models
from src.config import load_config, set_seed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Argument list; ``None`` means ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train one experiment configuration.")
    parser.add_argument("--config", required=True, help="Path to a YAML file in configs/.")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set train.learning_rate=0.0001",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed.")
    parser.add_argument("--author", default=None, help="Who runs it (default: $EXPERIMENT_AUTHOR or git user).")
    parser.add_argument("--notes", default="", help="Remark stored in the experiment log.")
    parser.add_argument("--no-log", action="store_true", help="Do not record this run in experiments/.")
    return parser.parse_args(argv)


def build_optimizer(train_cfg: dict) -> keras.optimizers.Optimizer:
    """Create the optimizer named in the config.

    Args:
        train_cfg: The ``train`` section of the config.

    Returns:
        A Keras optimizer.

    Raises:
        ValueError: If the optimizer name is unknown.
    """
    name = train_cfg["optimizer"]
    lr = train_cfg["learning_rate"]
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr)
    if name == "sgd_momentum":
        return keras.optimizers.SGD(learning_rate=lr, momentum=train_cfg["momentum"])
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    raise ValueError(f"Unknown optimizer {name!r}; use sgd, sgd_momentum or adam.")


def build_callbacks(train_cfg: dict) -> list[keras.callbacks.Callback]:
    """Create the training callbacks.

    Early stopping monitors the validation loss and restores the best weights.

    Args:
        train_cfg: The ``train`` section of the config.

    Returns:
        The list of callbacks passed to ``model.fit``.
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
        ),
    ]


def train(config: dict) -> tuple[keras.Model, keras.callbacks.History]:
    """Train one model on the train split and validate on the validation split.

    Args:
        config: A configuration loaded with :func:`src.config.load_config`.

    Returns:
        The trained model and its Keras training history.
    """
    set_seed(config["experiment"]["seed"])
    train_cfg = config["train"]

    datasets = data.make_datasets(config)
    model = models.build_model(config)
    model.compile(
        optimizer=build_optimizer(train_cfg),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.F1Score(average="macro", name="macro_f1")],
    )

    class_weight = data.compute_class_weights(config) if config["data"]["class_weights"] else None
    history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=train_cfg["epochs"],
        callbacks=build_callbacks(train_cfg),
        class_weight=class_weight,
    )
    return model, history


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point: load the config, train, and log the run.

    Every run is recorded in ``experiments/`` (run record, learning curves and
    the experiment table) through :func:`src.experiments.run_one`, unless
    ``--no-log`` is given.
    """
    args = parse_args(argv)
    if args.no_log:
        config = load_config(args.config, args.set)
        if args.seed is not None:
            config["experiment"]["seed"] = args.seed
        train(config)
        return

    from src.experiments import run_one

    run_one(args.config, overrides=args.set, seed=args.seed, author=args.author, notes=args.notes, train_fn=train)


if __name__ == "__main__":
    main()
