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

## Expected layout after download

```
data/raw/
├── HAM10000_metadata.csv      lesion_id, image_id, dx, dx_type, age, sex, localization
├── HAM10000_images_part_1/    5,000 JPEG images (600 x 450)
└── HAM10000_images_part_2/    5,015 JPEG images
```

Note: some mirrors (for example on Hugging Face) contain more than 10,015 rows or come pre-split. Use one of the two sources above.
