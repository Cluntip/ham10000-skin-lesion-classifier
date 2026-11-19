# HAM10000 Skin-Lesion Classifier

AIE231 Neural Networks course project (Alamein International University).
7-class classification of dermatoscopic skin-lesion images with TensorFlow/Keras:
a custom CNN trained from scratch, compared with ResNet-50 / VGG16 transfer learning.

> **Disclaimer:** this is a course project, not a medical diagnostic tool. Its predictions must not be used for clinical decisions.

---

## 1. Problem statement *(draft, to be finalised by the Tech Lead)*

**User.** A dermatologist or general practitioner who takes a dermatoscopic photo of a pigmented skin lesion during a check-up and wants a second opinion to decide which lesions to prioritise for biopsy or specialist referral.

**Prediction.** Given one dermatoscopic image, the model predicts which of 7 diagnostic categories the lesion belongs to, with a confidence for each class.

**Cost of a wrong prediction.** Errors are not equal. Missing a malignant lesion (melanoma `mel`, basal cell carcinoma `bcc`, actinic keratosis / intraepithelial carcinoma `akiec`) by calling it benign delays cancer treatment, which is the most expensive error. Calling a benign lesion (for example a nevus `nv`) malignant costs an unnecessary biopsy and patient anxiety, which is cheaper but not free.

| Item | Choice | Justification |
|---|---|---|
| Input | RGB image, resized to `224 × 224 × 3` (pixel values scaled) | Size expected by ImageNet-pretrained ResNet-50 / VGG16; the custom CNN may use a smaller size (decided by the Model Architect). |
| Output | Softmax over 7 classes | Single-label multi-class problem. |
| Loss | Categorical cross-entropy (with class weights) | Standard for softmax outputs; class weights counter the heavy imbalance. |
| Primary metric | **Macro-F1** | `nv` is ~67 % of the data, so accuracy is misleading (always predicting `nv` already scores ~67 %). Macro-F1 weights every class equally, including the rare dangerous ones. |
| Key secondary metric | **Melanoma recall** | The most costly error is a missed melanoma. |

## 2. Dataset

**HAM10000** ("Human Against Machine with 10000 training images"): dermatoscopic images of common pigmented skin lesions, collected from two clinical sites over 20 years.

- Official source: Harvard Dataverse, [doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)
- Mirror (convenient on Colab): [Kaggle: Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Licence: **CC BY-NC 4.0** (non-commercial use with attribution, which covers academic use)
- Paper: Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data, 5, 180161. [doi:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)

Reported class distribution (10,015 images; **to be verified in our EDA**):

| Code | Diagnosis | Images |
|---|---|---:|
| `nv` | Melanocytic nevus (benign) | 6,705 |
| `mel` | Melanoma (malignant) | 1,113 |
| `bkl` | Benign keratosis-like lesion | 1,099 |
| `bcc` | Basal cell carcinoma (malignant) | 514 |
| `akiec` | Actinic keratosis / intraepithelial carcinoma | 327 |
| `vasc` | Vascular lesion | 142 |
| `df` | Dermatofibroma | 115 |

Known data issues to check in the EDA:

- **Severe class imbalance**: `nv` is about 58× larger than `df`.
- **Several images of the same lesion** (shared `lesion_id`). Splits must keep every lesion inside one split, or the test set leaks.
- **Missing or unknown metadata** (`age`, `sex`), plus images taken with different devices and lighting.

See [`data/README.md`](data/README.md) for download instructions.

## 3. Team and ownership

| Member | Role | Owns |
|---|---|---|
| Ammar | Tech Lead + Baseline & Analysis Engineer | Problem formulation, repo structure, `configs/`, `src/config.py`, `src/train.py` entry point, merges, EDA notebook, baselines, final 3-seed results |
| Ghada | Data Engineer | `src/data.py`, `data/`, splits, preprocessing, imbalance handling, augmentation |
| Marwan | Model Architect | `src/models.py` (Models B and C), hand parameter counts, regularization, GradientTape step |
| Amr | Experimentation Engineer | Experiment runner, experiment log, learning-curve plots, LR / batch-size / optimizer studies |
| Youssef | Evaluation & Deployment Engineer | `src/evaluate.py`, `src/predict.py`, `app/`, error analysis |

## 4. Repository structure

```
ham10000-skin-lesion-classifier/
├── README.md            problem, results table, setup and run instructions
├── requirements.txt     pinned package versions
├── configs/             one YAML file per experiment (all extend base.yaml)
├── data/                download script and README (raw data is NOT committed)
├── notebooks/           01_eda.ipynb, 02_results.ipynb (they import from src/)
├── src/
│   ├── config.py        config loading, project paths, seeding
│   ├── data.py          loading, cleaning, splitting, preprocessing
│   ├── models.py        baselines and neural architectures
│   ├── train.py         training entry point, callbacks, logging
│   ├── evaluate.py      metrics, confusion matrix, ROC, error analysis
│   └── predict.py       inference on a raw image
├── experiments/         experiment log (CSV) and saved curves
├── models/              final saved model and preprocessing objects
├── app/                 demo application
├── report/              report PDF and slides
└── tests/               unit tests
```

## 5. Setup

```bash
git clone https://github.com/Cluntip/ham10000-skin-lesion-classifier.git
cd ham10000-skin-lesion-classifier
python -m venv .venv && source .venv/bin/activate   # skip on Colab
pip install -r requirements.txt
```

On Google Colab, select a GPU runtime, then:

```python
!git clone https://github.com/Cluntip/ham10000-skin-lesion-classifier.git
%cd ham10000-skin-lesion-classifier
!pip install -q -r requirements.txt
```

The versions in `requirements.txt` match the Colab runtime `2026.10` (Python 3.13, TensorFlow 2.21.0, NumPy 2.1.3). If Colab updates, check with `!pip list` before changing the pins.

## 6. Usage *(commands become runnable as the modules are implemented)*

```bash
python data/download.py                                    # download HAM10000 into data/raw/
python -m src.train --config configs/base.yaml             # train one configuration
python -m src.train --config configs/base.yaml --set train.learning_rate=0.0001 --seed 7
python -m src.evaluate --config configs/final.yaml         # test-set evaluation of the final model
python -m src.predict path/to/image.jpg                    # single-image prediction
python app/app.py                                          # live demo
pytest                                                     # unit tests
```

## 7. Results *(filled in Week 13; test-set numbers for the final model and the baselines)*

| Model | Macro-F1 | Melanoma recall | Accuracy | Params | Notes |
|---|---:|---:|---:|---:|---|
| Majority class (trivial baseline) | | | | 0 | |
| Logistic regression (learned baseline) | | | | | |
| Model A: custom CNN | | | | | |
| Model B: ResNet-50, frozen base | | | | | |
| Model C: ResNet-50, fine-tuned | | | | | |

Final model: mean ± std over 3 seeds.

## 8. Team workflow

1. Never push directly to `main`. Create a branch for each task: `git checkout -b <name>/<topic>` (e.g. `ghada/lesion-split`).
2. Commit small, meaningful changes from **your own GitHub account**.
3. Open a pull request. It must be reviewed and approved by a **different** member before it is merged.
4. Use the pull-request checklist (`.github/pull_request_template.md`).

## 9. Reproducibility rules

- Every run is started from a config file in `configs/` and logged in `experiments/`.
- Seeds are fixed with `src.config.set_seed` (Python, NumPy, TensorFlow).
- No hard-coded paths: build paths with `src.config.PROJECT_ROOT` / `resolve_path`.
- The **test set is used once**, at the end, for the final model and the baselines. All tuning and model selection use the validation set.
- Preprocessing is defined once in `src/data.py` and reused by training and inference.

## 10. References

1. P. Tschandl, C. Rosendahl and H. Kittler, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions," *Scientific Data*, vol. 5, 180161, 2018.
2. K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR*, 2016.
3. K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," *ICLR*, 2015.
