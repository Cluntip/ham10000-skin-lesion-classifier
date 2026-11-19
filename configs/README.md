# configs/

One YAML file per experiment configuration.

- `base.yaml` is the **reference configuration**. Every tuning study and ablation starts from it.
- An experiment file starts with `extends: base.yaml` and overrides **only** the keys it studies (see `example_lr_1e-4.yaml`). Nested keys are merged, so you never copy the whole base file.
- For quick one-off changes, use the command line instead of a new file:
  `python -m src.train --config configs/base.yaml --set train.batch_size=64`
- Loading and merging are done by `src/config.py` (`load_config`).

Naming convention: `<study>_<value>.yaml`, e.g. `lr_1e-3.yaml`, `bs_64.yaml`, `opt_sgd.yaml`, `abl_no_dropout.yaml`.
