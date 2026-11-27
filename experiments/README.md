# experiments/

The team's experiment log and learning curves (owner: Experimentation Engineer).
**Every training run by every member is logged here**, including the baselines.

```
experiments/
├── experiment_log.csv     the experiment table (one row per run), generated
├── runs/<run_id>.json     one record per run: config, history, metrics, timing
├── curves/<run_id>.png    learning curves of each run (train vs validation)
├── curves/study_*.png     curves of all runs of one study overlaid
├── summary_*.csv          mean ± std over seeds, generated
└── eval/                  outputs of src/evaluate.py (Evaluation & Deployment)
```

## How a run gets logged

You don't need to do anything special: both entry points log automatically.

```bash
python -m src.train --config configs/base.yaml                 # one run
python -m src.experiments run --study learning_rate            # a whole tuning study
python -m src.experiments run --config configs/base.yaml --seeds 42 43 44 --name final
```

Each run writes `runs/<run_id>.json` and `curves/<run_id>.png`, then rebuilds `experiment_log.csv`.

**On Colab, say who you are once per session** so the `author` column is right:

```python
%env EXPERIMENT_AUTHOR=Ghada
```

(or pass `--author Ghada`). Without it, the git user name is used.

**Then commit your run files** (`experiments/runs/`, `experiments/curves/`, `experiments/experiment_log.csv`) in your branch. Every run has its own JSON file, so two people's runs never conflict. If git reports a conflict in `experiment_log.csv`, it is a generated file, so rebuild it and commit:

```bash
python -m src.experiments log
```

## Columns of `experiment_log.csv`

| Column | Meaning |
|---|---|
| `run_id` | `<date>-<time>_<experiment name>_s<seed>`, also the JSON file name |
| `date`, `author` | when and by whom the run was made |
| `study`, `factor`, `factor_value` | tuning study / ablation, the config key it varies and its value in this run |
| `model`, `seed` | model name from the config, random seed |
| `optimizer`, `learning_rate`, `batch_size` | main hyperparameters (all of them are in `hyperparameters`) |
| `epochs_trained`, `best_epoch` | epochs run before early stopping; epoch with the lowest validation loss |
| `train_time_s` | wall-clock training time in seconds |
| `params_total`, `params_trainable` | parameter counts (same as `model.summary()`) |
| `train_*`, `val_*` | loss, accuracy and macro-F1 **at the best epoch** (the weights early stopping restores) |
| `fit_diagnosis` | `overfitting`, `underfitting / not converged` or `good fit` (rules in `src/curves.py`) |
| `generalization_gap` | validation loss minus training loss at the best epoch |
| `setup_hash` | fingerprint of the config without name and seed: same hash = same setup, different seed |
| `config_file`, `git_commit` | how to reproduce the run (`+dirty` = uncommitted changes) |
| `hyperparameters` | all model / training settings as JSON |
| `notes` | free-text remark |

Only **validation** metrics are logged. Test-set numbers come from `src/evaluate.py`, run once at the end.

## Logging a baseline (no Keras history)

```python
from src.config import load_config
from src.tracking import log_run

config = load_config("configs/base.yaml", ["model.name=logreg"])
log_run(config, metrics={"val_macro_f1": 0.41, "val_accuracy": 0.70},
        params={"total": n_params}, train_time_s=seconds, notes="learned baseline")
```

## Tuning studies

Defined in `configs/studies.yaml`; see `report/experimentation_plan.md` for the plan.

```bash
python -m src.experiments list                                     # studies and their values
python -m src.experiments run --study optimizer --dry-run          # show the runs without training
python -m src.experiments run --study optimizer --skip-existing    # resume after a disconnect
python -m src.experiments plot --study optimizer --metric val_loss
python -m src.experiments summary --study optimizer                # mean ± std over seeds
```
