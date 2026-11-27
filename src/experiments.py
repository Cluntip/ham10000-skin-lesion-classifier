"""Experiment runner: one-factor-at-a-time tuning studies, seeds and summaries.

The tuning plan lives in ``configs/studies.yaml``. Each study lists config
files that all extend ``configs/base.yaml`` (the reference configuration) and
change exactly one key, its *factor*. The runner checks this rule before it
trains anything, then trains every config of the study, logs each run with
:mod:`src.tracking` and saves its learning curves.

Usage::

    python -m src.experiments list
    python -m src.experiments run --study learning_rate            # 3 runs
    python -m src.experiments run --study optimizer --set model.name=resnet50_frozen
    python -m src.experiments run --config configs/base.yaml --seeds 42 43 44
    python -m src.experiments run --study batch_size --dry-run     # show the plan only
    python -m src.experiments log                                  # rebuild the CSV table
    python -m src.experiments plot --study learning_rate --metric val_loss
    python -m src.experiments summary                              # mean ± std over seeds

Only the train and validation splits are used. Model selection is done on
validation metrics; the test split belongs to ``src/evaluate.py``.

Owner: Experimentation Engineer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

import yaml

from src import tracking
from src.config import CONFIG_DIR, load_config, resolve_path
from src.curves import plot_study

#: File that defines the tuning studies.
STUDIES_FILE = CONFIG_DIR / "studies.yaml"

#: Keys that are allowed to differ between the configs of one study besides its factor.
BOOKKEEPING_KEYS = {"experiment.name"}


def load_studies(path: str | Path = STUDIES_FILE) -> dict[str, dict]:
    """Read the study definitions.

    Args:
        path: The studies YAML file.

    Returns:
        ``{study_name: {"factor": dotted_key, "configs": [file, ...]}}``;
        config file names are relative to the studies file's folder.
    """
    with open(resolve_path(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def study_config_paths(study: dict, studies_file: str | Path = STUDIES_FILE) -> list[Path]:
    """Return the absolute paths of the config files of one study."""
    folder = resolve_path(studies_file).parent
    return [folder / name for name in study["configs"]]


def check_one_factor(study_name: str, study: dict, studies_file: str | Path = STUDIES_FILE) -> None:
    """Check that the configs of a study differ only in the study's factor.

    This enforces one-factor-at-a-time tuning: if any other setting differs,
    a difference in the results could not be attributed to the factor.

    Args:
        study_name: Name of the study (for the error message).
        study: The study definition from :func:`load_studies`.
        studies_file: The studies YAML file (config paths are relative to it).

    Raises:
        ValueError: If a config changes a key other than the factor, or if
            the factor takes the same value twice.
    """
    factor = study["factor"]
    configs = {path.name: tracking.flatten(load_config(path)) for path in study_config_paths(study, studies_file)}
    reference_name, reference = next(iter(configs.items()))

    for name, flat in configs.items():
        changed = {
            key for key in set(reference) | set(flat)
            if reference.get(key) != flat.get(key)
        } - BOOKKEEPING_KEYS - {factor}
        if changed:
            raise ValueError(
                f"Study {study_name!r}: {name} differs from {reference_name} in {sorted(changed)}, "
                f"but only {factor!r} may change."
            )

    values = [flat.get(factor) for flat in configs.values()]
    if len(set(map(str, values))) != len(values):
        raise ValueError(f"Study {study_name!r}: {factor!r} repeats a value: {values}")


def already_logged(config: dict, study: str) -> bool:
    """Return True if a run with the same setup, seed and study is already logged.

    Used by ``--skip-existing`` to resume a study after a Colab disconnect.
    """
    target = (tracking.setup_hash(config), int(config["experiment"]["seed"]), study)
    for record in tracking.load_runs(tracking.experiments_dir(config)):
        if (record["setup_hash"], int(record["seed"]), record["study"]) == target:
            return True
    return False


def run_one(
    config_path: str | Path,
    *,
    overrides: Sequence[str] = (),
    seed: int | None = None,
    study: str = "",
    factor: str | None = None,
    author: str | None = None,
    notes: str = "",
    skip_existing: bool = False,
    train_fn: Callable | None = None,
) -> dict | None:
    """Train one configuration with one seed and log it.

    Args:
        config_path: The config file to train.
        overrides: ``dotted.key=value`` overrides applied on top of the file.
        seed: Seed for this run (default: ``experiment.seed`` of the config).
        study: Study name recorded in the log.
        factor: Dotted key the study varies (recorded with its value).
        author: Who runs it (see :func:`src.tracking.get_author`).
        notes: Free-text remark for the log.
        skip_existing: Skip the run if the same setup, seed and study is logged.
        train_fn: Function ``config -> (model, history)``; defaults to
            :func:`src.train.train`. Tests pass a fake one.

    Returns:
        The run record, or ``None`` if the run was skipped.
    """
    config = load_config(config_path, overrides)
    if seed is not None:
        config["experiment"]["seed"] = seed
    if study:
        config["experiment"]["study"] = study

    if skip_existing and already_logged(config, study):
        print(f"[skip] {Path(config_path).name} seed={config['experiment']['seed']} already logged for {study!r}")
        return None

    if train_fn is None:
        from src.train import train as train_fn  # imported here: TensorFlow loads only when training

    with tracking.Timer() as timer:
        model, history = train_fn(config)
    record = tracking.log_run(
        config,
        model=model,
        history=getattr(history, "history", history),
        train_time_s=timer.seconds,
        study=study,
        factor=factor,
        config_file=config_path,
        author=author,
        notes=notes,
    )
    summary = record["summary"]
    print(
        f"[done] {record['run_id']}: val_macro_f1={summary.get('val_macro_f1', float('nan')):.4f} "
        f"val_loss={summary.get('val_loss', float('nan')):.4f} ({record['fit_diagnosis']}, "
        f"{record['train_time_s']} s)"
    )
    return record


def plan_runs(args: argparse.Namespace) -> list[dict]:
    """Turn the ``run`` command's arguments into a list of planned runs.

    Args:
        args: Parsed arguments with ``study``, ``config`` and ``seeds``.

    Returns:
        One ``{"config_path", "seed", "study", "factor"}`` dict per run.
    """
    if args.study:
        studies = load_studies()
        if args.study not in studies:
            raise SystemExit(f"Unknown study {args.study!r}; choose from {sorted(studies)}.")
        study = studies[args.study]
        check_one_factor(args.study, study)
        paths, name, factor = study_config_paths(study), args.study, study["factor"]
    else:
        paths, name, factor = [resolve_path(args.config)], args.name or "", None

    runs = []
    for path in paths:
        seeds = args.seeds or [load_config(path, args.set)["experiment"]["seed"]]
        runs += [{"config_path": path, "seed": s, "study": name, "factor": factor} for s in seeds]
    return runs


def cmd_list(args: argparse.Namespace) -> None:
    """Print every study, its factor and the value each config gives it."""
    for name, study in load_studies().items():
        values = [tracking.get_by_path(load_config(p), study["factor"]) for p in study_config_paths(study)]
        print(f"{name:15s} factor={study['factor']:22s} values={values}")


def cmd_run(args: argparse.Namespace) -> None:
    """Train every planned run (or only print the plan with ``--dry-run``)."""
    runs = plan_runs(args)
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run['config_path'].name} seed={run['seed']} study={run['study'] or '-'}")
        if args.dry_run:
            continue
        run_one(
            run["config_path"],
            overrides=args.set,
            seed=run["seed"],
            study=run["study"],
            factor=run["factor"],
            author=args.author,
            notes=args.notes,
            skip_existing=args.skip_existing,
        )


def cmd_log(args: argparse.Namespace) -> None:
    """Rebuild ``experiments/experiment_log.csv`` from the run records."""
    out_dir = tracking.experiments_dir(load_config(args.config))
    table = tracking.build_log(out_dir)
    print(f"{len(table)} runs -> {out_dir / 'experiment_log.csv'}")


def cmd_plot(args: argparse.Namespace) -> None:
    """Overlay one metric for all runs of a study and save the figure."""
    out_dir = tracking.experiments_dir(load_config(args.config))
    runs = [r for r in tracking.load_runs(out_dir) if r["study"] == args.study and r.get("history")]
    if not runs:
        raise SystemExit(f"No logged runs with history for study {args.study!r}.")
    path = out_dir / "curves" / f"study_{args.study}_{args.metric}.png"
    plot_study(runs, metric=args.metric, title=f"{args.study}: {args.metric}", save_path=path)
    print(f"{len(runs)} runs -> {path}")


def cmd_summary(args: argparse.Namespace) -> None:
    """Print (and save) the mean ± std of the validation metrics over seeds."""
    out_dir = tracking.experiments_dir(load_config(args.config))
    table = tracking.build_log(out_dir)
    if args.study:
        table = table[table["study"] == args.study]
    if table.empty:
        raise SystemExit("No matching runs logged yet.")
    summary = tracking.seed_summary(table)
    path = out_dir / f"summary_{args.study or 'all'}.csv"
    summary.to_csv(path, index=False)
    readable = [c for c in summary.columns if not c.endswith(("_mean", "_std")) and c != "setup_hash"]
    print(summary[readable].to_string(index=False))
    print(f"-> {path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments of the runner."""
    parser = argparse.ArgumentParser(description="Run and summarise experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Show the tuning studies.").set_defaults(func=cmd_list)

    run = sub.add_parser("run", help="Train a study or one config (optionally with several seeds).")
    target = run.add_mutually_exclusive_group(required=True)
    target.add_argument("--study", help="Study name from configs/studies.yaml.")
    target.add_argument("--config", help="A single config file.")
    run.add_argument("--name", default="", help="Study/group name to record for a single --config run.")
    run.add_argument("--seeds", type=int, nargs="+", help="Seeds to run (default: the config's seed).")
    run.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Overrides for every run.")
    run.add_argument("--author", default=None, help="Who runs it (default: $EXPERIMENT_AUTHOR or git user).")
    run.add_argument("--notes", default="", help="Remark stored in the log.")
    run.add_argument("--skip-existing", action="store_true", help="Skip runs that are already logged.")
    run.add_argument("--dry-run", action="store_true", help="Only print the planned runs.")
    run.set_defaults(func=cmd_run)

    for name, func, help_text in (
        ("log", cmd_log, "Rebuild experiments/experiment_log.csv."),
        ("plot", cmd_plot, "Overlay the curves of one study."),
        ("summary", cmd_summary, "Mean ± std over seeds."),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", default="configs/base.yaml", help="Config that names the experiments folder.")
        if name in ("plot", "summary"):
            p.add_argument("--study", required=(name == "plot"), default=None, help="Study name.")
        if name == "plot":
            p.add_argument("--metric", default="val_loss", help="History key, e.g. val_loss or val_macro_f1.")
        p.set_defaults(func=func)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
