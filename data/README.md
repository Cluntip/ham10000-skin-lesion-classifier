# data/

The dataset is **not** stored in git (it is about 3 GB). Download it into `data/raw/`.

## Source and licence

- **HAM10000**, Harvard Dataverse: [doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)
- Kaggle mirror: [kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Licence: CC BY-NC 4.0 (non-commercial, attribution required)
- Cite: P. Tschandl, C. Rosendahl and H. Kittler, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions," *Scientific Data*, vol. 5, 180161, 2018.

## Download

```bash
python data/download.py
```

The downloader reads the public Kaggle archive and extracts the metadata and
original JPEGs into `data/raw/`. If the transfer is interrupted, rerun the same
command to resume from the images already saved. Run the audit after the
downloader reports that the dataset is ready.

## Expected layout after download

```
data/raw/
├── HAM10000_metadata.csv      lesion_id, image_id, dx, dx_type, age, sex, localization
├── HAM10000_images_part_1/    5,000 JPEG images (600 x 450)
└── HAM10000_images_part_2/    5,015 JPEG images
```

Note: some mirrors (for example on Hugging Face) contain more than 10,015 rows or come pre-split. Use one of the two sources above.

## Data engineer checks

Run the audit before training. It reads the extracted dataset in place, reports
metadata and image integrity, and prints split distributions and train-only
weights. Use a different extracted location with the existing path override.

```bash
python -m src.data --config configs/base.yaml
python -m src.data --set "data.raw_dir=/path/to/extracted/HAM10000"
```

From a fresh checkout, run these commands from the repository root to
download the dataset and generate `data/processed/ham10000_split_seed42.csv`.
The export contains `image_id`, `lesion_id`, `dx`, and `split`.

```bash
python -m pip install -r requirements.txt
python data/download.py
python -m src.data --config configs/base.yaml --save-split
```

Give sample audits a separate output directory to keep the full-dataset
manifest available.

```bash
python -m src.data --config configs/base.yaml --save-split --set data.raw_dir=data/processed/smoke data.processed_dir=data/processed/smoke
```

The pipeline removes exact repeated metadata rows. It retains rows with missing
optional demographic fields because the model uses images alone. Missing
required identifiers or labels, duplicate image IDs after deduplication,
ambiguous image filenames, conflicting lesion labels, and missing images require
investigation. The audit also decodes each matched image and reports unreadable
files. The source dataset stays intact.

Splitting balances per-diagnosis image counts against the configured ratios
while assigning each lesion as a group. It reserves a training lesion per
diagnosis and aims to represent each diagnosis in all splits when at least three
lesions are available for that diagnosis. Achieved ratios depend on lesion
sizes. The split is reproducible from the CSV and config. Experiment
configurations inherit `data.split.seed: 42` from `configs/base.yaml`, so
varying `experiment.seed` preserves the split assignments.

`src.data.make_datasets` returns lazy batched TensorFlow datasets keyed by
`train`, `val`, and `test`. Images are RGB, resized bilinearly to the
configured size (currently 224 x 224), and emitted as float32 in [0, 1].
Labels are one-hot in YAML order, `akiec, bcc, bkl, df, mel, nv, vasc`.
Training and inference both call `src.data.load_image` to decode files and
apply the same preprocessing. Confirm the configured size and the transfer model's input adapter with
the Model Architect. Train-only weights use N / (K * n_class) from the training
split. Train-only augmentation applies horizontal flips and rotations up to
9 degrees in either direction with reflection padding on the fly.

## Local findings

The Kaggle metadata has 10,015 image records across 7,470 lesions, with up to
six images per lesion. There were 57 missing ages, 57 unknown sex values, and
234 unknown localizations. There were zero missing required identifiers or
labels and zero duplicate rows or image IDs. The pipeline retains the
corresponding rows because the model takes images as input.

The seed-42 split assigned 7,011 train images across 5,228 lesions, plus 1,502
images across 1,121 lesions in each holdout. Image IDs and lesion IDs were
disjoint across the three splits. The class counts and resulting training
weights below were calculated from the full metadata.

| Diagnosis | Total images | Train | Validation | Test | Training weight |
|---|---:|---:|---:|---:|---:|
| akiec | 327 | 229 | 49 | 49 | 4.3737 |
| bcc | 514 | 360 | 77 | 77 | 2.7821 |
| bkl | 1,099 | 769 | 165 | 165 | 1.3024 |
| df | 115 | 81 | 17 | 17 | 12.3651 |
| mel | 1,113 | 779 | 167 | 167 | 1.2857 |
| nv | 6,705 | 4,693 | 1,006 | 1,006 | 0.2134 |
| vasc | 142 | 100 | 21 | 21 | 10.0157 |

The full dataset audit matched all 10,015 JPEGs to their metadata. There were
zero unmatched files or metadata rows. All 10,015 images decoded as
600 × 450 RGB JPEGs.

## Ablations

Use the same model, split, seed, and training settings across these runs.
Compare B with A for class weighting and C with A for augmentation. D adds
a comparison with both enabled. Run after the baselines and CNN are available.

| Run | data.class_weights | data.augmentation |
|---|---|---|
| A | false | false |
| B | true | false |
| C | false | true |
| D | true | true |

Example invocation for run A:

```bash
python -m src.train --config configs/base.yaml --set data.class_weights=false data.augmentation=false
```

Log the run ID, date, author, model, configuration, seed, elapsed time,
parameter count, and training and validation metrics in the team experiment
log. Compare validation macro-F1, accuracy, and per-class precision, recall,
and F1. Keep the test split for final evaluation. The report should include
the observed audit and ablation results, the work-package contribution, and
where AI coding assistance was used.
