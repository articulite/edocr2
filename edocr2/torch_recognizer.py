"""The recognizer (keras_ocr's CRNN) in PyTorch, so it runs on the GPU and trains there.

The network is the Keras one layer for layer, including its quirks, so the Keras weights carry
over and the outputs agree (see tests/test_torch_recognizer.py):
  - the input is turned so that the image's width is the sequence (Keras: Permute, then a flip
    of the height axis);
  - batch normalisation uses Keras' epsilon, 1e-3;
  - the spatial transformer samples with its own bilinear interpolation, ported as written;
  - each "backward" LSTM reads the sequence reversed and its output stays reversed, as Keras'
    go_backwards leaves it, before it is added to or set beside the forward one;
  - the first two time steps are dropped and the rest decoded greedily (CTC).

    rec = TorchRecognizer.load("edocr2/models/recognizer_dimensions_2.pt")
    rec.recognize(crop) -> "Ø12 H7"
"""

import os

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HEIGHT, WIDTH, DISCARD = 31, 200, 2


class CRNN(nn.Module):
    def __init__(self, classes, height=HEIGHT, width=WIDTH, filters=(64, 128, 256, 256, 512, 512, 512), units=128):
        super().__init__()
        self.height, self.width = height, width
        f = filters
        self.convs = nn.ModuleList([nn.Conv2d(i, o, 3, padding=1) for i, o in zip((1,) + f[:-1], f)])
        self.bns = nn.ModuleDict({str(k): nn.BatchNorm2d(f[k], eps=1e-3) for k in (2, 4, 6)})   # after conv 3, 5, 7
        steps, across = width // 4, height // 4
        # the spatial transformer's localisation network
        self.loc1 = nn.Conv2d(f[-1], 16, 5, padding=2)
        self.loc2 = nn.Conv2d(16, 32, 5, padding=2)
        self.loc3 = nn.Linear(steps * across * 32, 64)
        self.loc4 = nn.Linear(64, 6)
        self.fc9 = nn.Linear(across * f[-1], units)
        self.lstm10, self.lstm10b = nn.LSTM(units, units, batch_first=True), nn.LSTM(units, units, batch_first=True)
        self.lstm11, self.lstm11b = nn.LSTM(units, units, batch_first=True), nn.LSTM(units, units, batch_first=True)
        self.fc12 = nn.Linear(2 * units, classes)

    @staticmethod
    def _transform(x, theta):
        """keras_ocr.recognition._transform. x: (B, H, W, C) channels last; theta: (B, 6)."""
        b, h, w, c = x.shape
        theta = theta.view(b, 2, 3)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device)
        ys = torch.linspace(-1.0, 1.0, h, device=x.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.ones(h * w, device=x.device)])   # (3, H*W)
        t = theta @ grid                                                                            # (B, 2, H*W)
        px = 0.5 * (t[:, 0].reshape(-1) + 1.0) * w
        py = 0.5 * (t[:, 1].reshape(-1) + 1.0) * h
        x0, y0 = px.floor().long(), py.floor().long()
        x1, y1 = x0 + 1, y0 + 1
        x0, x1 = x0.clamp(0, w - 1), x1.clamp(0, w - 1)
        y0, y1 = y0.clamp(0, h - 1), y1.clamp(0, h - 1)
        base = (torch.arange(b, device=x.device) * h * w).repeat_interleave(h * w)
        flat = x.reshape(-1, c)
        va, vb = flat[base + y0 * w + x0], flat[base + y1 * w + x0]
        vc, vd = flat[base + y0 * w + x1], flat[base + y1 * w + x1]
        x0f, x1f, y0f, y1f = x0.float(), x1.float(), y0.float(), y1.float()
        wa = ((x1f - px) * (y1f - py)).unsqueeze(1)
        wb = ((x1f - px) * (py - y0f)).unsqueeze(1)
        wc = ((px - x0f) * (y1f - py)).unsqueeze(1)
        wd = ((px - x0f) * (py - y0f)).unsqueeze(1)
        return (wa * va + wb * vb + wc * vc + wd * vd).view(b, h, w, c)

    def forward(self, images):
        """images: (B, height, width) floats in [0, 1]. Returns (B, steps - 2, classes) softmax."""
        x = images.transpose(1, 2).flip(2).unsqueeze(1)          # (B, 1, width, height): Permute, then flip
        for k, conv in enumerate(self.convs):
            x = F.relu(conv(x))
            if str(k) in self.bns:
                x = self.bns[str(k)](x)
                if k in (2, 4):
                    x = F.max_pool2d(x, 2)
        nhwc = x.permute(0, 2, 3, 1)                              # (B, steps, across, C)
        loc = F.relu(self.loc2(F.relu(self.loc1(x))))
        loc = F.relu(self.loc3(loc.permute(0, 2, 3, 1).reshape(x.shape[0], -1)))
        x = self._transform(nhwc, self.loc4(loc))
        x = F.relu(self.fc9(x.reshape(x.shape[0], x.shape[1], -1)))
        r1 = self.lstm10(x)[0] + self.lstm10b(x.flip(1))[0]
        x = torch.cat([self.lstm11(r1)[0], self.lstm11b(r1.flip(1))[0]], dim=2)
        return F.softmax(self.fc12(x), dim=2)[:, DISCARD:]


def keras_to_state(recognizer):
    """A CRNN state dict from a keras_ocr Recognizer with its weights loaded (needs TensorFlow)."""
    import keras
    m = recognizer.model
    get = lambda name: [np.asarray(w) for w in m.get_layer(name).get_weights()]
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
    state = {}
    for k in range(7):
        w, b = get(f"conv_{k + 1}")
        state[f"convs.{k}.weight"], state[f"convs.{k}.bias"] = t(w.transpose(3, 2, 0, 1)), t(b)
    for k, name in ((2, "bn_3"), (4, "bn_5"), (6, "bn_7")):
        g, be, mean, var = get(name)
        state.update({f"bns.{k}.weight": t(g), f"bns.{k}.bias": t(be), f"bns.{k}.running_mean": t(mean),
                      f"bns.{k}.running_var": t(var), f"bns.{k}.num_batches_tracked": torch.tensor(0)})
    loc = next(l for l in m.layers if isinstance(l, keras.Model))
    convs = [l for l in loc.layers if isinstance(l, keras.layers.Conv2D)]
    dense = [l for l in loc.layers if isinstance(l, keras.layers.Dense)]
    for name, layer in zip(("loc1", "loc2"), convs):
        w, b = layer.get_weights()
        state[f"{name}.weight"], state[f"{name}.bias"] = t(w.transpose(3, 2, 0, 1)), t(b)
    for name, layer in zip(("loc3", "loc4", "fc9", "fc12"), dense + [m.get_layer("fc_9"), m.get_layer("fc_12")]):
        w, b = layer.get_weights()
        state[f"{name}.weight"], state[f"{name}.bias"] = t(w.T), t(b)
    for name, keras_name in (("lstm10", "lstm_10"), ("lstm10b", "lstm_10_back"), ("lstm11", "lstm_11"), ("lstm11b", "lstm_11_back")):
        kernel, recurrent, bias = get(keras_name)
        # same gate order (input, forget, cell, output); Keras has one bias, PyTorch two
        state[f"{name}.weight_ih_l0"], state[f"{name}.weight_hh_l0"] = t(kernel.T), t(recurrent.T)
        state[f"{name}.bias_ih_l0"], state[f"{name}.bias_hh_l0"] = t(bias), torch.zeros(len(bias))
    return state


class TorchRecognizer:
    """Recognizer.recognize, in PyTorch, plus a batched version."""

    def __init__(self, alphabet, state=None, device=None):
        # full float32 on the GPU, so results match the CPU's (TF32 moved outputs by ~3e-3)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.alphabet = alphabet
        self.blank_label_idx = len(alphabet)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = CRNN(len(alphabet) + 1).to(self.device).eval()
        if state is not None:
            self.net.load_state_dict(state)

    @classmethod
    def load(cls, path, device=None):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        return cls(saved["alphabet"], saved["state"], device)

    def save(self, path):
        torch.save({"alphabet": self.alphabet, "state": self.net.state_dict()}, path)

    @staticmethod
    def prepare(image):
        """A crop as the Keras recognizer takes it: letterboxed to 31x200 on black, grey, 0-1."""
        from edocr2.keras_ocr import tools
        fitted = tools.read_and_fit(filepath_or_array=image, width=WIDTH, height=HEIGHT, cval=0)
        if fitted.ndim == 3:
            fitted = cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY)
        return fitted.astype("float32") / 255

    def decode(self, probs):
        out = []
        for row in probs.argmax(-1).tolist():
            chars, last = [], None
            for i in row:
                if i != last and i != self.blank_label_idx:
                    chars.append(self.alphabet[i])
                last = i
            out.append("".join(chars))
        return out

    @torch.no_grad()
    def probabilities(self, images):
        x = torch.from_numpy(np.stack([self.prepare(i) for i in images])).to(self.device)
        return self.net(x).cpu().numpy()

    def recognize_batch(self, images):
        """Text of each crop; empty crops read as no text."""
        good = [i for i, im in enumerate(images) if isinstance(im, np.ndarray) and im.size and min(im.shape[:2]) >= 2]
        out = [""] * len(images)
        if good:
            for i, text in zip(good, self.decode(self.probabilities([images[i] for i in good]))):
                out[i] = text
        return out

    def recognize(self, image):
        return self.recognize_batch([image])[0]


def convert(keras_path, out_path=None):
    """Write the .pt beside a .keras recognizer (needs TensorFlow, once)."""
    from edocr2.keras_ocr.recognition import Recognizer
    from edocr2.tools.ocr_pipelines import read_alphabet
    alphabet = read_alphabet(keras_path)
    rec = Recognizer(alphabet=alphabet, weights=None)
    rec.model.load_weights(keras_path)
    out_path = out_path or os.path.splitext(keras_path)[0] + ".pt"
    TorchRecognizer(alphabet, keras_to_state(rec), "cpu").save(out_path)
    return out_path
