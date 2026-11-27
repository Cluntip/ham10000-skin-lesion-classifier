# Evaluation and deployment plan

Owner: Youssef (Evaluation & Deployment Engineer). Draft for checkpoint 1; feeds report sections 6 (Results and error analysis) and 7 (Deployment).

## 1. Metrics

| Metric | Role | Why |
|---|---|---|
| **Macro-F1** | Primary; used for model selection on the validation set | `nv` is ~67 % of the images, so a model that always says `nv` gets ~67 % accuracy. Macro-F1 gives every class the same weight, including rare dangerous ones. |
| **Melanoma recall** | Key secondary | A missed melanoma is the most expensive error. |
| **Malignant recall** | Secondary | Share of `akiec` / `bcc` / `mel` lesions predicted as *any* malignant class. This answers the screening question "would this lesion be referred?". Calling a melanoma a BCC still gets a biopsy. |
| Accuracy, balanced accuracy | Reported | Required by the spec. Shown next to the majority baseline to show why accuracy misleads here. |
| Per-class precision / recall / F1, confusion matrix | Reported | Show *which* classes fail. |
| One-vs-rest ROC + AUC per class, macro AUC | Reported | Measures ranking quality without depending on the threshold. |

All of these come from `src.evaluate.compute_metrics`.

## 2. Which split is used for what

- **Validation split:** all model comparisons, tuning and ablations. It also gets the confusion matrix, ROC and failure cases for **every** model (A, B, C and the baselines), as the role requires. Command: `python -m src.evaluate --config <cfg> --model <file>` (`--split val` is the default).
- **Test split:** used **once**, at the end, for the final model (3 seeds) and the two baselines. It has to be asked for explicitly with `--split test`. No decision is changed after looking at the test results.
- Every image of one lesion stays in one split (`group_by: lesion_id`), so test images are never near-duplicates of training images.

## 3. Interfaces I depend on (to agree with the team)

| With | What I need |
|---|---|
| Ghada (data) | `make_datasets(config)["val"/"test"]` is **not shuffled** (✅ in her pipeline). `evaluate.py` takes its metadata from the same cleaned splits (`data._prepared_splits`) and stops if the label order differs. Training and inference share `data.load_image` / `data.preprocess_image`; `predict.py` calls them directly. |
| Ammar / Marwan (models, training) | Models output a **softmax over the 7 classes in `data.classes` order**. Save the final model with `src.predict.save_artifacts(model, config)`. |
| Ammar (baselines) | The learned baseline saves its validation probabilities as a `.npy` array of shape `(n_val, 7)` in the row order of the validation split. `python -m src.evaluate --config configs/base.yaml --probs <file>.npy --name logreg_baseline` then produces the same metrics and plots as for the neural models. The majority baseline is built into `evaluate.py` (`--baseline majority`): it uses the training-set class frequencies for every image. |
| Amr (experiments) | `metrics.json` from each evaluation can be added to the experiment log. |

## 4. Error analysis protocol

1. Run `evaluate.py` on the final model. It writes `worst_errors.csv`, the 20 most confident mistakes with their probabilities and metadata, and `worst_errors.png`, a gallery of those images.
2. Look at each image and fill in the `failure_category` and `explanation` columns by hand. The planned categories are:
   - **Look-alike classes**, e.g. `mel` vs `nv`, or `bkl` vs `mel`. These are visually similar even for dermatologists.
   - **Image quality / artefacts**: hair, ruler marks, ink, vignetting, blur or bad lighting.
   - **Rare class**: `df` and `vasc` have only about 100 to 140 images each.
   - **Possible label noise**: diagnoses confirmed by `consensus` or `follow_up` instead of `histo`. Check this with `error_rate_by(metadata, ..., "dx_type")`.
   - **Unusual pattern**: an atypical body site or age, checked with `error_rate_by(..., "localization")`.
3. Take the most common confusion from `top_confusions.csv` and propose one fix. Candidates are a higher class weight for the missed class, targeted augmentation, or a lower decision threshold for `mel` chosen on the validation set. Test the fix on the validation set if there is time.

## 5. Deployment

- **Artefacts:** `models/final_model.keras` plus `models/inference_config.json`. See `models/README.md`.
- **Entry point:** `python -m src.predict image.jpg` returns the class, the confidence, all 7 probabilities and the total malignant probability.
- **Demo:** a Gradio app (`python app/app.py`) with top-3 classes, a low-confidence warning and a referral note. It includes example images, one of them a known failure case.
- **Measurements** (from `python -m src.predict img.jpg --benchmark 50`):
  - Inference time per image: mean ± std after 5 warm-up runs, measured on both the Colab CPU and the Colab GPU.
  - Model size on disk in MB.
- **First measurements:** Colab T4, 26 Nov 2026. The model was untrained but has the final ResNet-50 size, and the image was 600×450 like the HAM10000 images. These numbers must be re-measured with the final model.

  | Version | Inference time per image (mean ± std, 50 runs) | Model size |
  |---|---:|---:|
  | Eager Keras call `model(x)` | 414.8 ± 186.5 ms | 95.1 MB |
  | Graph compiled once with `tf.function` and a fixed input signature (current) | **32.8 ± 9.7 ms** (≈12× faster) | 95.1 MB |

  - The times include reading the JPEG and preprocessing it, not only the model.
  - The first prediction takes about 7.5 s, because loading the model and tracing the graph happen then. The demo calls `warm_up()` at start-up, so the first user doesn't wait.
  - The model file is close to GitHub's 100 MB limit, so ship it as a `v1.0` release asset.
  - **Note for the Tech Lead:** the Colab runtime on 26 Nov had keras 3.13.2 and numpy 2.4.6, but `requirements.txt` pins keras 3.12.0 and numpy 2.1.3.
- **Bonus candidates:**
  - A public demo on Hugging Face Spaces.
  - A TFLite version with float16 / int8 quantisation, reporting size, speed and the change in macro-F1.
  - The unit tests (`tests/`) run in GitHub Actions CI.

## 6. Timeline

| Checkpoint | My deliverable |
|---|---|
| 1 (Week 9) ✅ | This plan; `evaluate.py` (all metrics, plots, error tables) with unit tests; `predict.py` (save/load, prediction, benchmark, warm-up); the Gradio demo; the artefact contract. Everything tested on a Colab T4: all tests pass, and the demo launched with a dummy model. |
| 2 (Week 10) ✅ (learned baseline pending) | `evaluate.py` connected to Ghada's pipeline. Majority baseline measured on the validation set; logistic-regression baseline measured as soon as Ammar's probabilities exist. Comparison table in `experiments/eval/summary_val.csv`. |
| 3 (Week 12) | Confusion matrix, ROC and failure cases for Models A, B and C on the validation set. Demo running with a temporary model. |
| 4 (Week 13) | One test-set run for the final model and the baselines; error analysis filled in; benchmark numbers; report sections 6 and 7 drafted. |
| Week 14 | Live demo, including a failure case. |

## 7. Baseline results (validation split)

Measured on 27 Nov 2026 on Colab with `notebooks/03_baseline_evaluation.ipynb`. HAM10000 has 10,015 images and 7,470 lesions. The validation split has 1,502 images and 1,121 lesions, and no image or lesion appears in two splits (checked by Ghada's audit). The test split was not used.

| Model | Macro-F1 | Melanoma recall | Malignant recall | Accuracy | Balanced acc. | Macro AUC |
|---|---:|---:|---:|---:|---:|---:|
| Majority class (always `nv`) | 0.115 | 0.000 | 0.000 | 0.670 | 0.143 | 0.500 |
| Logistic regression (learned baseline) | waiting for Ammar's `experiments/logreg_baseline_val.npy` | | | | | |

**What this shows:**
- Predicting `nv` for every image already gives **67 % accuracy**, but it misses **every** melanoma and every malignant lesion, and its macro-F1 is only 0.115. This is the evidence for choosing macro-F1 and melanoma recall over accuracy. Every neural model must beat **0.115 macro-F1** and **0 % melanoma recall**.
- A balanced accuracy of 0.143 is exactly 1/7, and an AUC of 0.5 is chance level. Both are what a model that never looks at the image should get, which is a sanity check that `evaluate.py` is correct.
- The validation and test splits have the same class counts (1,006 `nv`, 167 `mel`, 165 `bkl`, 77 `bcc`, 49 `akiec`, 21 `vasc`, 17 `df`). Rare classes have few validation images (`df` = 17, `vasc` = 21), so their per-class recall will be noisy. Report it with that caveat.

Reproduce with `python -m src.evaluate --config configs/base.yaml --baseline majority`.
