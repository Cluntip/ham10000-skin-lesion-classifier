# models/

The **final** saved model and everything `src/predict.py` needs (owner: Evaluation & Deployment Engineer).

## Contents

| File | What it is | Written by |
|---|---|---|
| `final_model.keras` | Trained Keras model; input = output of `src.data.preprocess_image`, output = softmax over the 7 classes | `src.predict.save_artifacts` |
| `inference_config.json` | The full config the model was trained with (`data.image_size`, `data.classes`, ...). Inference passes it to `src.data.preprocess_image`, so preprocessing is identical to training | `src.predict.save_artifacts` |

Save the final model with:

```python
from src.predict import save_artifacts
save_artifacts(model, config)          # writes models/final_model.keras + models/inference_config.json
```

There are no separate scaler or encoder files: image preprocessing has no fitted parameters (resize + pixel scaling), and the class order is in the config. If the Data Engineer adds a fitted object later (for example per-channel mean / std computed on the training split), it is saved here too and loaded by `load_artifacts`.

## Large files

GitHub rejects files larger than 100 MB, and a ResNet-50 model is close to that. If the final model is too large, attach it to the `v1.0` GitHub release (or use Git LFS) and document the download command here.
