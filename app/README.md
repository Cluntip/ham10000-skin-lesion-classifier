# app/

Live demo built on top of `src/predict.py` (owner: Evaluation & Deployment Engineer).
It uses the same model and preprocessing as `python -m src.predict`.

```bash
pip install -r requirements.txt -r app/requirements.txt
python app/app.py            # http://127.0.0.1:7860
python app/app.py --share    # temporary public link (useful on Colab)
```

The demo needs the final model in `models/` (see [`models/README.md`](../models/README.md)).

What it shows for one uploaded image:

- the top-3 classes with their probabilities,
- a **low-confidence warning** when the top probability is below 50 %,
- the total probability of a malignant class (`akiec`, `bcc`, `mel`) and a "would be referred" note above 30 %,
- the inference time.

`app/examples/` holds a few test-set images for the demo, including at least one **known failure case** from the error analysis (the rubric asks the demo to show one). Only add images from the test split, and keep them small.
