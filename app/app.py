"""Live demo: upload a dermatoscopic image, get the predicted lesion class.

Built on :func:`src.predict.predict_array`, so the demo uses exactly the same
model and preprocessing as ``python -m src.predict``.

Run::

    pip install -r app/requirements.txt
    python app/app.py                 # local, http://127.0.0.1:7860
    python app/app.py --share         # temporary public link (e.g. from Colab)

Owner: Evaluation & Deployment Engineer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

# Make `src` importable when the app is started as `python app/app.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_array, warm_up  # noqa: E402

CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma (pre-malignant)",
    "bcc": "Basal cell carcinoma (malignant)",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma (benign)",
    "mel": "Melanoma (malignant)",
    "nv": "Melanocytic nevus (benign mole)",
    "vasc": "Vascular lesion (benign)",
}

#: Below this top-class probability the demo says the model is unsure.
LOW_CONFIDENCE = 0.50
#: Above this total malignant probability the demo recommends a specialist.
REFERRAL_THRESHOLD = 0.30

DISCLAIMER = (
    "**Course project (AIE231), not a medical device.** "
    "Predictions must not be used for diagnosis or treatment decisions."
)

EXAMPLES_DIR = Path(__file__).parent / "examples"


def classify(image):
    """Gradio callback: run the model on one uploaded image.

    Args:
        image: ``uint8`` NumPy array of shape ``(h, w, 3)`` from ``gr.Image``.

    Returns:
        The class probabilities for ``gr.Label`` and a Markdown explanation.
    """
    if image is None:
        return {}, "Upload an image first."
    result = predict_array(image)
    label = result["label"]

    notes = [f"### {label}: {CLASS_DESCRIPTIONS.get(label, label)}", f"Confidence: **{result['confidence']:.1%}**"]
    if result["confidence"] < LOW_CONFIDENCE:
        notes.append("⚠️ **Low confidence**: the model is unsure between several classes; treat this result as unreliable.")
    notes.append(f"Total probability of a malignant class (akiec, bcc, mel): **{result['malignant_probability']:.1%}**")
    if result["malignant_probability"] >= REFERRAL_THRESHOLD:
        notes.append("🔎 In a real screening setting this lesion would be **referred to a specialist**.")
    notes.append(f"<sub>Inference time: {result['inference_ms']:.0f} ms</sub>")
    return result["probabilities"], "\n\n".join(notes)


def build_demo() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks() as demo:
        gr.Markdown("# Skin-lesion classifier (HAM10000)\n" + DISCLAIMER)
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="numpy", label="Dermatoscopic image")
                button = gr.Button("Classify", variant="primary")
            with gr.Column():
                probabilities = gr.Label(num_top_classes=3, label="Top-3 classes")
                explanation = gr.Markdown()
        examples = sorted(str(p) for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if examples:
            gr.Examples(examples=examples, inputs=image, label="Examples (including a known failure case)")
        button.click(classify, inputs=image, outputs=[probabilities, explanation])
    return demo


def main() -> None:
    """Command-line entry point: start the demo server."""
    parser = argparse.ArgumentParser(description="Start the Gradio demo.")
    parser.add_argument("--share", action="store_true", help="Create a temporary public link.")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    print(f"Model loaded and warmed up in {warm_up() / 1000:.1f} s")
    build_demo().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
