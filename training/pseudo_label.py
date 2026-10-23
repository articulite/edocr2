"""Crops in eDOCr2's own drawing style, labelled by the released dimension recognizer.

    python training/pseudo_label.py DRAWING [...] --out DATA_DIR/pseudo [--dpi 200,300]

Fine-tuning on drawgen sheets taught the recognizer new lettering and characters, and on the
paper's drawings it began to misread a little (4x⌀6.60 as 4xX6.0). These crops hold it to what
it read before on drawings like those. Only use drawings outside the benchmark: the repository's
tests/*.PDF (Aufspannung, Aufspannung_Ecke, Elevator_bottom) are not among tests/test_samples.

Every crop the pipeline hands the dimension recognizer is kept with the released model's
reading, when that reading has a digit (the pipeline keeps those as dimensions) and is longer
than one character. Writes DATA_DIR/pseudo/dim/<n>.png and labels.jsonl, as collect_crops does.
"""

import argparse, json, os, re, sys
from pathlib import Path

import cv2
import numpy as np
import pypdfium2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from edocr2 import cli          # noqa: E402


def pages(path, dpis):
    if path.lower().endswith(".pdf"):
        doc = pypdfium2.PdfDocument(path)
        for dpi in dpis:
            for page in doc:
                yield cv2.cvtColor(np.array(page.render(scale=dpi / 72, grayscale=True).to_pil()), cv2.COLOR_GRAY2BGR)
    else:
        yield cv2.imread(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drawings", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", default="200,300")
    args = ap.parse_args()
    cli.use_tesseract()
    models = cli.load_models(None)            # the released recognizers
    rec = models[1]
    out = Path(args.out) / "dim"
    out.mkdir(parents=True, exist_ok=True)
    kept = []
    read = rec.recognize
    # the released model's own slips are not worth keeping: stray colons, a reading that starts
    # or ends on a separator, two separators in a row
    looks_wrong = re.compile(r":|^[.,]|[.,]$|[.,][.,]|[.,][^0-9]")
    def keep(image):
        text = read(image)
        if (any(ch.isdigit() for ch in text) and len(text) > 1 and not looks_wrong.search(text)
                and image.size and min(image.shape[:2]) >= 2):
            kept.append((image.copy(), text))
        return text
    rec.recognize = keep
    for d in args.drawings:
        for img in pages(d, [int(x) for x in args.dpi.split(",")]):
            cli.read_page(img, models, fixes=())
    with open(out / "labels.jsonl", "w", encoding="utf-8") as f:
        for n, (img, text) in enumerate(kept, 1):
            cv2.imwrite(str(out / f"{n}.png"), img)
            f.write(json.dumps({"file": f"{n}.png", "label": text, "source": "released model"}, ensure_ascii=False) + "\n")
    print(f"{len(kept)} crops")


if __name__ == "__main__":
    main()
