"""The detector and both recognizers as ONNX, for running in a browser (onnxruntime-web).

    python -m edocr2 export-onnx --out DIR

writes craft.onnx, recognizer_gdts.onnx, recognizer_dimensions_2.onnx and alphabets.json (each
recognizer's alphabet and the routing alphabet, see cli.load_models). Every output is checked
against PyTorch on a test input before it is kept. The detector takes (1, 3, H, W) normalised
RGB (craft.compute_input) and gives (1, H/2, W/2, 2); a recognizer takes (N, 31, 200) grey 0-1
strips (TorchRecognizer.prepare) and gives (N, 48, classes) probabilities.
"""

import json, os

import numpy as np
import torch

OPSET = 17


def _export(module, x, path, outputs, dynamic_axes, tolerance=1e-3):
    """Export, then check onnxruntime against PyTorch. The reference is taken first: the
    exporter leaves the module changed (its outputs moved by up to 0.3 after an export)."""
    import onnxruntime as ort
    with torch.no_grad():
        ref = module(x).numpy()
    torch.onnx.export(module, x, path, input_names=["input"], output_names=outputs, opset_version=OPSET, dynamo=False,
                      dynamic_axes=dynamic_axes)
    y = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(y - ref).max())
    if diff > tolerance:
        raise RuntimeError(f"{os.path.basename(path)} differs from PyTorch by {diff:.1e}")
    return diff


def export(out, folder=None):
    from edocr2 import cli
    from edocr2.keras_ocr import craft, tools
    from edocr2.torch_recognizer import TorchRecognizer, HEIGHT, WIDTH
    os.makedirs(out, exist_ok=True)

    cfg = craft.PRETRAINED_WEIGHTS[("clovaai_general", True)]
    det = craft.build_torch_model(tools.download_and_verify(url=cfg["url"], filename=cfg["filename"], sha256=cfg["sha256"])).eval()

    class Maps(torch.nn.Module):     # the score maps only, not the features
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)[0]

    path = os.path.join(out, "craft.onnx")
    # a smooth input: on noise, ties in the maps make any two runtimes look apart
    x = torch.from_numpy(np.tile(np.linspace(-2, 2, 320, dtype="float32"), (1, 3, 256, 1)))
    diff = _export(Maps(det), x, path, ["maps"], {"input": {2: "h", 3: "w"}, "maps": {1: "h2", 2: "w2"}})
    print(f"craft.onnx: {os.path.getsize(path) / 1e6:.0f} MB, max diff {diff:.1e}")

    folder = folder or cli.model_folder()
    alphabets = {}
    for name in ("recognizer_gdts", "recognizer_dimensions_2"):
        rec = TorchRecognizer.load(os.path.join(folder, name + ".pt"), device="cpu")
        alphabets[name] = rec.alphabet
        path = os.path.join(out, name + ".onnx")
        # a blank strip with one bar: the transformer's sampling turns noise into rounding ties
        x = torch.zeros(4, HEIGHT, WIDTH)
        x[:, 10:20, 40:160] = 1
        diff = _export(rec.net, x, path, ["probs"], {"input": {0: "n"}, "probs": {0: "n"}})
        print(f"{name}.onnx: {os.path.getsize(path) / 1e6:.0f} MB, max diff {diff:.1e}")
    routing = alphabets["recognizer_dimensions_2"]
    if len(routing) > len(cli.ROUTING_BASE):
        routing = cli.ROUTING_BASE + "XTUIN↧⌴⌵□×"
    alphabets["routing"] = routing
    with open(os.path.join(out, "alphabets.json"), "w", encoding="utf-8") as f:
        json.dump(alphabets, f, ensure_ascii=False, indent=1)
    print("alphabets.json")
