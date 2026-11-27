# Experimentation plan: tuning, tracking and learning curves

Work package 5, Experimentation Engineer (Amr). Spec sections 3.4 (optimizer from evidence) and 3.5 (experiment tracking, hyperparameter tuning, ablations, reporting).

## 1. Scope

| Deliverable | Where |
|---|---|
| Experiment log: every run of every member in one table | `src/tracking.py`, `experiments/experiment_log.csv` |
| Experiment runner: one-factor-at-a-time studies, multiple seeds | `src/experiments.py`, `configs/studies.yaml` |
| Learning-curve plots and over/underfitting diagnosis | `src/curves.py`, `experiments/curves/` |
| Learning-rate, batch-size and optimizer studies | `configs/lr_*.yaml`, `configs/bs_*.yaml`, `configs/opt_*.yaml` |
| Report section 5 (experimental setup, hyperparameter studies) | this document, then the report |

## 2. Experiment log

Each run is saved as its own JSON record in `experiments/runs/<run_id>.json`. The record holds the full config, the per-epoch history, the metrics at the best epoch, parameter counts, training time, author, seed and git commit. `experiments/experiment_log.csv` is generated from the records.

Why one file per run: team members train in separate Colab sessions. If everyone appended rows to one CSV, every push would conflict. Separate files never conflict, and the CSV can always be regenerated (`python -m src.experiments log`).

The columns cover everything spec 3.5 asks for: run ID, date, author, model, hyperparameters, random seed, training time, parameters, and training / validation metrics. They are documented in [`experiments/README.md`](../experiments/README.md).

**Which numbers are logged.** Early stopping monitors the validation loss and restores the best weights, so the logged metrics are those of the **best epoch** (lowest validation loss), i.e. of the weights the model actually keeps. Only training and validation metrics are logged; the test set is used once, at the end, by `src/evaluate.py`.

## 3. Reference configuration and one-factor-at-a-time rule

All studies start from `configs/base.yaml`:

| Setting | Reference value |
|---|---|
| Model | `custom_cnn` (Model A) |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Max epochs / early-stopping patience | 30 / 5 (on validation loss) |
| Class weights, augmentation | on, on |
| Seed | 42 |

Each study varies **one** config key (its *factor*) and keeps everything else equal to the reference, so any change in the results can be attributed to that factor. The runner enforces this rule: before training, `check_one_factor` compares every config of the study with the first one and stops if any key other than the factor differs.

Every study includes `base.yaml` itself, so each study has its own reference run. With the same seed and deterministic ops (`set_seed`), the repeated reference runs should give the same result, which is a free reproducibility check.

## 4. Tuning studies

| Study | Factor | Values | Why these values |
|---|---|---|---|
| Learning rate | `train.learning_rate` | 0.01, **0.001**, 0.0001 | One decade either side of Adam's default: too high (unstable or diverging loss), default, too low (slow convergence). |
| Batch size | `train.batch_size` | 16, **32**, 64 | Powers of two around the reference. Smaller batches give noisier gradients (a mild regulariser) and more updates per epoch but slower epochs; larger batches are faster per epoch but may generalise worse. |
| Optimizer | `train.optimizer` | SGD, SGD + momentum 0.9, **Adam** | The three optimizers the spec requires (section 3.4). Momentum smooths SGD's updates; Adam adapts the step size per parameter. |
| Architecture factor | to be agreed with the Model Architect | 3 values | Required by spec 3.5: for example the number of filters of the custom CNN, or the units of the classifier head. |

Bold = reference value. Runs per study: 3 configs × 1 seed.

**Expected interaction.** All optimizers use the reference learning rate (0.001), because only one factor changes at a time. Plain SGD usually needs a larger learning rate than Adam, so it may look slow here. That is a finding to explain in the report, not a reason to change the protocol.

**Which model is tuned.** The studies run on the reference model in `base.yaml` (Model A). If the final model is Model B or C, the learning-rate study is repeated on it (`--set model.name=resnet50_finetune`), because fine-tuning a pretrained network needs a much smaller learning rate than training from scratch.

**Compute budget.** The spec limits one run to 30 minutes on a free Colab GPU, so the three studies (9 runs) take at most about 4.5 GPU-hours. They can be split across sessions: `--skip-existing` resumes a study after a disconnect without repeating finished runs.

## 5. Selection rule

1. For each study, choose the value with the highest **validation macro-F1** (the project's primary metric).
2. If two values are within one standard deviation of each other (once several seeds exist), prefer the cheaper or simpler one (larger batch, fewer parameters).
3. Combine the chosen values into the final configuration. The Tech Lead trains it with seeds 42, 43 and 44.
4. The test set is never used for any of these decisions.

## 6. Learning curves and fit diagnosis

Every run saves `experiments/curves/<run_id>.png` (training vs validation loss and macro-F1, best epoch marked). `python -m src.experiments plot --study <name>` overlays all runs of a study.

Each run also gets a first label in the log (`src/curves.py`, `diagnose_fit`):

| Label | Rule | Typical remedy |
|---|---|---|
| underfitting / not converged | validation loss still at its minimum on the last epoch | train longer, larger learning rate, more capacity |
| overfitting | after the best epoch, validation loss rose by more than 10 % while training loss kept falling | more dropout / augmentation, earlier stopping, smaller model |
| good fit | otherwise | — |

The label is a starting point. The important curves are still inspected and discussed in the report.

## 7. Final result over three seeds

```bash
python -m src.experiments run --config configs/final.yaml --seeds 42 43 44 --name final
python -m src.experiments summary --study final
```

The summary reports the mean ± sample standard deviation (`ddof=1`) of every validation metric over the three seeds (`experiments/summary_final.csv`).

## 8. Interfaces with the team

| Member | Interface |
|---|---|
| Ammar (Tech Lead) | `python -m src.train` now logs every run automatically (`--no-log` turns it off). The final 3-seed run and its mean ± std use the commands in section 7. Baselines are logged with `tracking.log_run(config, metrics=..., params=...)`. |
| Ghada (Data) | The batch-size study relies on `make_datasets` reading `train.batch_size` from the config (it does). The imbalance and augmentation ablations can be written as studies in `configs/studies.yaml` (factor `data.class_weights` or `data.augmentation`), so they get the same one-factor check, logging and plots. |
| Marwan (Models) | Same for the architecture and regularization ablations (e.g. factor `model.dropout`). Who runs the architecture-factor tuning study is still to be agreed. |
| Youssef (Evaluation) | Evaluating a logged model with `--name <run_id>` links `experiments/eval/<run_id>_val/` to its row in the log. `experiments/eval/` stays owned by Evaluation & Deployment. |
| Everyone | On Colab, run `%env EXPERIMENT_AUTHOR=<name>` once, then commit the new files in `experiments/runs/` and `experiments/curves/`. |

## 9. Timeline

| Milestone | Work | Status |
|---|---|---|
| 1 (Week 9) | Log schema, run IDs, timer, parameter counting, tuning plan | ✅ done |
| 2 (Week 10) | `src/tracking.py`, `src/curves.py`, `src/experiments.py` with the one-factor check, study configs, logging in `src/train.py`, unit tests | ✅ done |
| 3 (Weeks 11–12) | Run the learning-rate, batch-size and optimizer studies as soon as Model A trains end to end; first learning curves | waiting for Model A in `src/models.py` |
| 4 (Week 13) | Finish the studies and the architecture factor, choose the values on validation, support the 3-seed final run, write report section 5 | |
| Week 14 | Present the experiments section | |
