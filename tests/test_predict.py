"""End-to-end test of src/predict.py with a tiny dummy model (needs TensorFlow).

Skipped automatically where TensorFlow is not installed.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")

from src import data, predict  # noqa: E402

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """Save a tiny model + config, and use a simple resize if data.py is not implemented yet."""
    config = {"data": {"image_size": [32, 32], "classes": CLASSES}}

    def simple_preprocess(image, config):
        return tf.image.resize(tf.cast(image, tf.float32) / 255.0, config["data"]["image_size"])

    try:
        data.preprocess_image(tf.zeros((8, 8, 3), tf.uint8), config)
    except NotImplementedError:
        monkeypatch.setattr(data, "preprocess_image", simple_preprocess)

    model = keras.Sequential(
        [keras.Input((32, 32, 3)), keras.layers.GlobalAveragePooling2D(), keras.layers.Dense(7, activation="softmax")]
    )
    predict.load_artifacts.cache_clear()
    predict.compiled_forward.cache_clear()
    return predict.save_artifacts(model, config, tmp_path / "models")


def test_predict_image_returns_valid_probabilities(model_dir, tmp_path):
    image_path = tmp_path / "lesion.png"
    tf.io.write_file(str(image_path), tf.io.encode_png(tf.zeros((45, 60, 3), tf.uint8)))

    result = predict.predict_image(image_path, model_dir)
    assert result["label"] in CLASSES
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert result["confidence"] == max(result["probabilities"].values())
    assert 0.0 <= result["malignant_probability"] <= 1.0


def test_predict_array_matches_predict_image(model_dir, tmp_path):
    image = np.random.default_rng(0).integers(0, 255, (45, 60, 3), dtype=np.uint8)
    image_path = tmp_path / "lesion.png"
    tf.io.write_file(str(image_path), tf.io.encode_png(image))

    from_file = predict.predict_image(image_path, model_dir)["probabilities"]
    from_array = predict.predict_array(image, model_dir)["probabilities"]
    for name in CLASSES:
        assert from_file[name] == pytest.approx(from_array[name], abs=1e-6)


def test_warm_up_then_prediction_reuses_the_graph(model_dir, tmp_path):
    predict.warm_up(model_dir)
    image = np.zeros((300, 400, 3), dtype=np.uint8)  # different size from the warm-up image
    predict.predict_array(image, model_dir)
    # One trace only: the input signature is fixed, so new image sizes do not re-trace.
    assert predict.compiled_forward(model_dir).experimental_get_tracing_count() == 1


def test_missing_model_gives_clear_error(tmp_path):
    predict.load_artifacts.cache_clear()
    with pytest.raises(FileNotFoundError):
        predict.load_artifacts(tmp_path)
