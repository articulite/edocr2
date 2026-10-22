"""Training crops for the recognizers, cut by the pipeline itself from drawgen sheets and labelled
from each sheet's record.json.

    python training/collect_crops.py TRAIN_TUPLES_DIR --out DATA_DIR [--variants clean,scan]

Every crop the pipeline hands a recognizer is kept, as it was handed over (cut, split,
re-padded, turned), with where it came from on the sheet:
  - GD&T cells: the frame's record gives the cells in order (symbol, value as drawn - with
    the diameter sign, a decimal comma, the circled M - then the datums). A block whose cell
    count differs is skipped.
  - Dimension crops: labelled with the stretch of the covering callout's text that the crop
    reads closest to. The pipeline boxes words apart, so a crop is usually one or two words of
    a callout; a stretch is a run of its words (split at spaces and at the slash of limits),
    joined without spaces since the recognizer writes none. A crop no stretch comes close to
    is dropped, not guessed.
Writes DATA_DIR/{dim,gdt}/<n>.png and DATA_DIR/{dim,gdt}/labels.jsonl.
"""

import argparse, json, os, re, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from benchmarks.drawgen import GLYPH, bbox, cover, render, scan          # noqa: E402
from edocr2 import cli, tools                                           # noqa: E402
from training.alphabets import DIMENSIONS, GDT                          # noqa: E402

P = tools.ocr_pipelines


def drawn_value(c, style):
    """A frame's value cell as drawgen draws it (callout.rs, Kind::Control)."""
    s = f"{c['tolerance']:.3f}".rstrip("0").rstrip(".")
    s = ("⌀" if c.get("diameter_zone") else "") + s
    if style.get("decimal_comma"):
        s = s.replace(".", ",")
    return s + ("Ⓜ" if c.get("mmc") else "")


def tokens(text, split_slash=True):
    """The words of a callout as the recognizer would write them; limits written a/b are drawn
    stacked, so they are split at the slash unless split_slash is False (deviations in brackets
    are drawn on one line)."""
    t = text.replace("Ø", "⌀").replace("∅", "⌀")
    return [p for w in t.split() for p in (w.split("/") if split_slash else [w]) if p]


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def best_stretch(pred, words, alphabet):
    best = None
    for i in range(len(words)):
        for j in range(i + 1, min(len(words), i + 4) + 1):
            cand = "".join(words[i:j])
            if any(ch not in alphabet for ch in cand):
                continue
            d = edit(pred, cand)
            if best is None or d < best[0]:
                best = (d, cand)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tuples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="clean,scan")
    ap.add_argument("--shard", default="0/1", help="i/n: take every n-th sheet from the i-th, to run n at once into n folders")
    args = ap.parse_args()
    shard, shards = (int(v) for v in args.shard.split("/"))
    cli.use_tesseract()
    models = cli.load_models(None)
    rec_gdt, rec_dim = models[0], models[1]
    out = Path(args.out)
    for kind in ("dim", "gdt"):
        (out / kind).mkdir(parents=True, exist_ok=True)
    files = {k: open(out / k / "labels.jsonl", "a", encoding="utf-8") for k in ("dim", "gdt")}
    counts = {k: len(list((out / k).glob("*.png"))) for k in ("dim", "gdt")}
    kept, dropped = {"dim": 0, "gdt": 0}, {"dim": 0, "gdt": 0}

    # No Tesseract here: it is started three or four times per text box only to route boxes and
    # to read tables and notes, none of which yields a recognizer crop. Every box goes to the
    # recognizer (with the wider alphabet that is where THRU, MIN, MAX belong); crops no callout
    # matches are dropped by the labelling anyway.
    P.ocr_img_cv2 = lambda img, language=None, psm=11: ([], "")
    P.Pipeline.dimension_criteria = lambda self, img: True

    state = {}
    seg = tools.layer_segm.segment_img
    def segment(img, **kw):
        r = seg(img, **kw)
        state["frame"] = r[1]
        return r
    tools.layer_segm.segment_img = segment
    post = P.postprocess_detection
    def postprocess(img, box, *a, **kw):
        state["box"] = np.asarray(box, dtype=float)
        return post(img, box, *a, **kw)
    P.postprocess_detection = postprocess
    rgdt = P.recognize_gdt
    def recognize_gdt(img, block, recognizer):
        state["block"], state["cells"] = block, []
        return rgdt(img, block, recognizer)
    P.recognize_gdt = recognize_gdt
    gdt_read, dim_read = rec_gdt.recognize, rec_dim.recognize
    def read_gdt(image):
        state.setdefault("cells", []).append(image.copy())
        return gdt_read(image)
    def read_dim(image):
        text = dim_read(image)
        if state.get("box") is not None:
            f = state["frame"]
            state["dims"].append((image.copy(), state["box"] + [f.x, f.y], text))
        return text
    rec_gdt.recognize, rec_dim.recognize = read_gdt, read_dim
    blocks_done = []
    rgdt_inner = P.recognize_gdt
    def recognize_gdt_keep(img, block, recognizer):
        r = rgdt_inner(img, block, recognizer)
        blocks_done.append((block, list(state.get("cells", []))))
        return r
    P.recognize_gdt = recognize_gdt_keep

    root = Path(args.tuples)
    tuples = sorted(p.parent for p in root.rglob("record.json"))
    for n, t in enumerate(tuples):
        if n % shards != shard:
            continue
        record = json.loads((t / "record.json").read_text(encoding="utf-8"))
        style = record["style"]
        for variant in args.variants.split(","):
            gray = render(t / "drawing.pdf")
            warp = None
            if variant == "scan":
                gray, warp = scan(gray, seed=5000 + n)
            k = gray.shape[1] / record["sheet"]["w"]
            def sheet_box(c):
                b = [v * k for v in c["page_frame"]]
                if warp is not None:
                    pts = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
                    b = bbox(cv2.transform(pts[None], warp)[0])
                return b
            state.clear()
            state["dims"] = []
            blocks_done.clear()
            try:
                cli.read_page(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), models)
            except Exception as e:
                print(f"{t.name} {variant}: {e!r}", flush=True)
                continue
            controls = [(c, sheet_box(c)) for c in record["callouts"] if c["kind"] == "control"]
            for block, cells in blocks_done:
                rects = list(block)   # recognize_gdt gets the block's cells, sorted
                bb = [min(r.x for r in rects), min(r.y for r in rects), max(r.x + r.w for r in rects), max(r.y + r.h for r in rects)]
                match = next((c for c, b in controls if cover(b, bb) > 0.5), None)
                if match is None:
                    continue
                labels = [GLYPH[match["characteristic"]], drawn_value(match, style)] + list(match.get("datums") or [])
                if len(labels) != len(cells):
                    dropped["gdt"] += len(cells)
                    continue
                for img, label in zip(cells, labels):
                    if img.size == 0 or any(ch not in GDT for ch in label):
                        continue
                    counts["gdt"] += 1
                    cv2.imwrite(str(out / "gdt" / f"{counts['gdt']}.png"), img)
                    files["gdt"].write(json.dumps({"file": f"{counts['gdt']}.png", "label": label, "style": style["name"], "variant": variant}, ensure_ascii=False) + "\n")
                    kept["gdt"] += 1
            texts = [(c, sheet_box(c)) for c in record["callouts"] if c.get("text") and c["kind"] not in ("note", "control", "datum", "target")]
            for img, box, pred in state["dims"]:
                if img.size == 0 or min(img.shape[:2]) < 2:
                    continue
                bb = bbox(box)
                near = [c for c, b in texts if cover(bb, [b[0] - 15, b[1] - 15, b[2] + 15, b[3] + 15]) > 0.6]
                best = None
                for c in near:
                    for split in (True, False):
                        s = best_stretch(pred, tokens(c["text"], split), DIMENSIONS)
                        if s and (best is None or s[0] < best[0]):
                            best = s
                # the crop lies inside the callout, so the place already says which callout it
                # is; the reading only has to say which words. Half the label may be wrong:
                # THRU read as tHR0, by a recognizer that has no T or U, is two edits.
                if best is None or best[0] > max(1, (len(best[1]) + 1) // 2):
                    dropped["dim"] += 1
                    continue
                counts["dim"] += 1
                cv2.imwrite(str(out / "dim" / f"{counts['dim']}.png"), img)
                files["dim"].write(json.dumps({"file": f"{counts['dim']}.png", "label": best[1], "read": pred, "style": style["name"], "variant": variant}, ensure_ascii=False) + "\n")
                kept["dim"] += 1
        if (n + 1) % 10 == 0:
            print(f"{n + 1}/{len(tuples)} sheets; kept {kept}, dropped {dropped}", flush=True)
    print(f"done: kept {kept}, dropped {dropped}")


if __name__ == "__main__":
    main()
