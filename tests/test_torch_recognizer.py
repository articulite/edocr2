"""The PyTorch recognizer is the Keras one: same outputs from the same weights. Needs the models
(edocr2.cmd setup) and TensorFlow; skipped without them."""

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "edocr2", "models")
NAMES = ["recognizer_dimensions_2", "recognizer_gdts"]
have_models = all(os.path.isfile(os.path.join(MODELS, n + ".keras")) for n in NAMES)


def crops():
    """Text strips like the pipeline cuts: dark text on white, various sizes and lengths."""
    out = []
    for i, s in enumerate(["25.4", "Ø12 H7", "4x M6", "0.05", "R1.5", "+0.02 -0.01", "45°", "A", "120.00", "(58)"]):
        img = np.full((40 + 6 * (i % 3), 30 + 22 * len(s), 3), 255, np.uint8)
        cv2.putText(img, s.replace("Ø", "D").replace("°", "o"), (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9 + 0.1 * (i % 3), (0, 0, 0), 2)
        out.append(img)
    return out


@pytest.mark.skipif(not have_models, reason="models not set up")
@pytest.mark.parametrize("name", NAMES)
def test_pytorch_matches_keras(name, tmp_path):
    pytest.importorskip("tensorflow")
    from edocr2.keras_ocr.recognition import Recognizer
    from edocr2.tools.ocr_pipelines import read_alphabet
    from edocr2.torch_recognizer import TorchRecognizer, convert

    keras_path = os.path.join(MODELS, name + ".keras")
    alphabet = read_alphabet(keras_path)
    k = Recognizer(alphabet=alphabet, weights=None)
    k.model.load_weights(keras_path)
    t = TorchRecognizer.load(convert(keras_path, str(tmp_path / (name + ".pt"))), device="cpu")
    images = crops()
    x = np.stack([t.prepare(i) for i in images])[..., None]
    assert np.abs(k.model.predict(x, verbose=0) - t.probabilities(images)).max() < 1e-4
    assert [k.recognize(i) for i in images] == t.recognize_batch(images)


def test_the_pytorch_path_needs_no_tensorflow():
    code = ("import sys; from edocr2.keras_ocr.craft import TorchDetector; from edocr2.torch_recognizer import TorchRecognizer;"
            "from edocr2 import tools, textlayer; print('tensorflow' in sys.modules)")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root,
                         env={**os.environ, "PYTHONPATH": root}).stdout.strip()
    assert out == "False"
