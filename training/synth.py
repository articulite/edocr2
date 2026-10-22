"""Synthetic text strips for the recognizers: short dimension-like strings, rendered dark on light
in several fonts, roughened like a print or a scan. They keep the recognizer reading the fonts
it was first trained on (eDOCr2's own, bundled in edocr2/tools/*_fonts) while it learns new
characters and drawgen's, and they give every added character plenty of examples.
"""

import glob
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")


def fonts():
    """(text font, symbol font) pairs: the text font draws letters and digits, the symbol font
    the glyphs it lacks."""
    # one of each: on Windows *.ttf and *.TTF match the same files
    own = sorted({os.path.normcase(p): p for p in glob.glob(os.path.join(ROOT, "edocr2", "tools", "dimension_fonts", "*.[tT][tT][fF]"))}.values())
    extra = [os.path.join(WINDOWS, f) for f in ("arial.ttf", "segoeui.ttf", "calibri.ttf", "consola.ttf")]
    liberation = os.environ.get("EDOCR2_EXTRA_FONT")   # e.g. drawgen's LiberationSans-Regular.ttf
    text = [f for f in own + extra + ([liberation] if liberation else []) if f and os.path.isfile(f)]
    symbol = [f for f in (os.path.join(WINDOWS, "seguisym.ttf"), os.path.join(ROOT, "edocr2", "tools", "dimension_fonts", "Symbola.ttf")) if os.path.isfile(f)]
    return text, symbol


def number(rng, comma=False):
    whole = rng.choice([rng.randint(0, 9), rng.randint(10, 99), rng.randint(100, 999)])
    s = f"{whole}" if rng.random() < 0.3 else f"{whole}.{rng.randint(0, 999):0{rng.choice([1, 2, 3])}d}"
    return s.replace(".", ",") if comma else s


def dimension_text(rng):
    """A fragment such as the detector cuts from a dimension or a hole note."""
    comma = rng.random() < 0.2
    n = lambda: number(rng, comma)
    pick = rng.random()
    if pick < 0.12:
        return f"{rng.randint(2, 30)}X"
    if pick < 0.25:
        return f"⌀{n()}"
    if pick < 0.35:
        return rng.choice(["THRU", "MIN", "MAX", "MIN,", "TYP", "REF", "EQSP", "CSK", "CBORE", "X", "SR", "SØ"[:1] + "R"])
    if pick < 0.45:
        thread = f"M{rng.choice([3, 4, 5, 6, 8, 10, 12, 16, 20, 24])}"
        if rng.random() < 0.4:
            thread += rng.choice(["x", "X", "×"]) + rng.choice(["0.5", "0.7", "0.75", "1", "1.25", "1.5", "2"])
        if rng.random() < 0.5:
            thread += "-" + rng.choice(["6H", "5H", "7H", "6G", "6g", "6h", "4g6g"])
        return thread
    if pick < 0.55:
        return rng.choice(["H7", "H8", "h6", "g6", "f7", "js6", "k6", "p6", "r6", "s7", "e8", "d9", "c11", "b11", "u6", "v6", "y6", "z6"])
    if pick < 0.65:
        return rng.choice(["↧", "⌴", "⌵", "□"]) + (n() if rng.random() < 0.6 else "")
    if pick < 0.72:
        return rng.choice(["+", "-", "±"]) + n()
    if pick < 0.78:
        return f"({n()})" if rng.random() < 0.7 else f"[{n()}]"
    if pick < 0.84:
        return f"{rng.randint(1, 179)}°" + (f"{rng.randint(0, 59)}'" if rng.random() < 0.3 else "")
    if pick < 0.88:
        return f"R{n()}"
    if pick < 0.91:
        return rng.choice(["%", "#", "*", "×"]) + (n() if rng.random() < 0.5 else "")
    return n()


def gdt_text(rng):
    pick = rng.random()
    if pick < 0.3:
        return rng.choice("⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰")
    if pick < 0.5:
        return rng.choice("ABCD")
    v = f"{rng.choice([0.005, 0.01, 0.02, 0.025, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]):.3f}".rstrip("0").rstrip(".")
    if rng.random() < 0.25:
        v = v.replace(".", ",")
    if rng.random() < 0.35:
        v = "⌀" + v
    if rng.random() < 0.15:
        v += rng.choice("ⓂⓁⓅⓈⓉⓊⒺⒻ")
    return v


def render(text, rng, text_fonts, symbol_fonts):
    size = rng.randint(22, 48)
    tf = ImageFont.truetype(rng.choice(text_fonts), size)
    sf = ImageFont.truetype(rng.choice(symbol_fonts), size)
    # the text font for what it has, the symbol font for the rest (letters, digits, punctuation)
    plain = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,+-()[]:/'\"%#*=°±")
    pieces = [(ch, tf if ch in plain else sf) for ch in text]
    w = int(sum(f.getlength(ch) for ch, f in pieces)) + 2 * size
    img = Image.new("L", (w, int(size * 1.9)), 255)
    d = ImageDraw.Draw(img)
    x = size // 2 + rng.randint(0, size // 2)
    for ch, f in pieces:
        d.text((x, size * 0.35), ch, font=f, fill=0)
        x += f.getlength(ch)
    a = np.array(img)
    ys, xs = np.where(a < 128)
    if len(xs) == 0:
        return None
    pad = rng.randint(2, 8)
    a = a[max(0, ys.min() - pad): ys.max() + pad, max(0, xs.min() - pad): xs.max() + pad]
    if rng.random() < 0.5:   # print or scan: blur, noise, one bit
        a = cv2.GaussianBlur(a, (3, 3), rng.uniform(0.3, 1.0))
        a = np.clip(a.astype(np.int16) + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(0, 18), a.shape), 0, 255).astype(np.uint8)
        if rng.random() < 0.5:
            a = cv2.threshold(a, rng.randint(140, 200), 255, cv2.THRESH_BINARY)[1]
    if rng.random() < 0.2:   # thin or heavy strokes
        k = np.ones((2, 2), np.uint8)
        a = cv2.erode(a, k) if rng.random() < 0.5 else cv2.dilate(a, k)
    return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)


class Synth:
    """Endless (image, label) pairs for one recognizer ("dim" or "gdt")."""

    def __init__(self, kind, alphabet, seed=0):
        self.kind, self.alphabet, self.rng = kind, alphabet, random.Random(seed)
        self.text_fonts, self.symbol_fonts = fonts()

    def __call__(self):
        while True:
            text = dimension_text(self.rng) if self.kind == "dim" else gdt_text(self.rng)
            text = text.replace("Ø", "⌀")
            if any(ch not in self.alphabet for ch in text):
                continue
            img = render(text, self.rng, self.text_fonts, self.symbol_fonts)
            if img is not None:
                return img, text
