"""Score the pipeline on drawgen sheets, whose record.json gives every callout's text and its
place on the sheet. Each callout is judged only by what was read at that place.

    python benchmarks/drawgen.py TUPLES_DIR --out DIR [--variants clean,scan] [--keras-detector]

TUPLES_DIR holds drawgen tuples (each a folder with drawing.pdf and record.json). Each drawing is
rendered at 300 dpi, clean and "scanned" (slightly rotated, blurred, noisy, thresholded to one
bit, with a fixed seed per sheet), read, and scored. Results: DIR/<variant>/<sheet>.json per
sheet and DIR/summary.json.

Scored per callout (drawgen record schema 12+):
  frame         value, symbol and datums of the GD&T frame found at the callout's place
  thread        nominal size (M6), pitch when given, count prefix (4X)
  size          nominal, fit class, deviations or limits
  location      nominal
  finish        Ra value
  note          the note's words
  hole          kinds the record marks as hole callouts: size or limits, THRU or depth
Text read in pieces at one place is joined left to right before it is judged.
"""

import argparse, json, os, random, re, sys, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GLYPH = {"position": "⌖", "flatness": "⏥", "straightness": "⏤", "circularity": "○", "cylindricity": "⌭",
         "line_profile": "⌒", "surface_profile": "⌓", "perpendicularity": "⏊", "angularity": "∠",
         "parallelism": "⫽", "symmetry": "⌯", "concentricity": "◎", "circular_runout": "↗", "total_runout": "⌰"}


# ───────────────────────── rendering ─────────────────────────

def render(pdf, dpi=300):
    import pypdfium2
    page = pypdfium2.PdfDocument(str(pdf))[0]
    return np.array(page.render(scale=dpi / 72, grayscale=True).to_pil())


def scan(gray, seed):
    rng = random.Random(seed)
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-0.8, 0.8), 1.0)
    a = cv2.warpAffine(gray, m, (w, h), borderValue=255)
    a = cv2.GaussianBlur(a, (3, 3), rng.uniform(0.6, 1.2))
    a = np.clip(a.astype(np.int16) + np.random.default_rng(seed).normal(0, 14, a.shape), 0, 255).astype(np.uint8)
    return cv2.threshold(a, rng.randint(150, 190), 255, cv2.THRESH_BINARY)[1], m


# ───────────────────────── scoring ─────────────────────────

def norm(s):
    s = s.upper().replace(",", ".").replace("×", "X").replace("Ø", "⌀").replace("∅", "⌀")
    return re.sub(r"\s+", "", s)


def nums(t):
    return {f"{t:g}", f"{t:.2f}", f"{t:.3f}"} | {v.lstrip("0") for v in (f"{t:g}", f"{t:.2f}", f"{t:.3f}")}


def has_number(text, t):
    return any(norm(v) in norm(text) for v in nums(t))


def bbox(pts):
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    return [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]


def cover(a, b):
    """Share of box a covered by box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1.0, (a[2] - a[0]) * (a[3] - a[1]))


def place(box, items, grow=0.15):
    """Items read at a callout's place: lying mostly inside it, or covering most of it."""
    pad = grow * (box[3] - box[1])
    g = [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad]
    hit = [i for i in items if cover(i["bbox"], g) >= 0.6 or cover(box, i["bbox"]) >= 0.5]
    return sorted(hit, key=lambda i: (i["bbox"][0]))


def judge(c, here, frames_here):
    """{check: bool} for one callout."""
    k, text = c["kind"], " ".join(i["text"] for i in here)
    out = {}
    if k == "control":
        f = next((g for g in frames_here if has_number(g["text"], c["tolerance"])), frames_here[0] if frames_here else None)
        cells = [norm(x) for x in re.split(r"[|\n]", f["text"])] if f else []
        out["frame found"] = f is not None
        out["value"] = f is not None and any(re.sub(r"^⌀|[ⒺⒻⓁⓂⓅⓈⓉⓊ]+$", "", x) in {norm(v) for v in nums(c["tolerance"])} for x in cells[1:])
        out["symbol"] = f is not None and GLYPH[c["characteristic"]] in cells[0]
        if c.get("datums"):
            out["datums"] = [x for x in cells[1:] if re.fullmatch(r"[A-Z](-[A-Z])?", x)] == c["datums"]
        out["fully right"] = all(out.values())
        return out
    truth = (c.get("text") or "").replace(",", ".")
    # a thread, or a size written as a hole note (count, ending, counterbore)
    if k == "thread" or (k == "size" and re.search(r"THRU|↧|^\d+X", truth)):
        m = re.search(r"M(\d+(?:\.\d+)?)(?:\s*[X×x]\s*(\d+(?:\.\d+)?))?", truth)
        if k == "thread" and m:
            out["size"] = norm("M" + m.group(1)) in norm(text)
            if m.group(2):
                out["pitch"] = norm(f"M{m.group(1)}X{m.group(2)}") in norm(text)
        cnt = re.match(r"(\d+)\s*[X×]", truth)
        if cnt:
            out["count"] = norm(cnt.group(1) + "X") in norm(text)
        for word in ("THRU", "MIN", "MAX"):
            if word in truth:
                out[word] = word in norm(text)
        depth = re.search(r"↧\s*([\d.]+)", truth)
        if depth:
            out["depth"] = has_number(text, float(depth.group(1)))
        cb = re.search(r"⌴\s*Ø([\d.]+)", truth)
        if cb:
            out["counterbore Ø"] = has_number(text, float(cb.group(1)))
    if k in ("size", "location", "hole") and c.get("size"):
        s = c["size"]
        out["nominal"] = has_number(text, s["nominal"])
        if c.get("fit"):
            out["fit class"] = norm(c["fit"]["class"]) in norm(text)
        hi, lo = s["nominal"] + (s.get("upper") or 0), s["nominal"] - (s.get("lower") or 0)
        if hi != lo and has_number(truth, hi) and has_number(truth, lo):
            # written as limits, larger over smaller
            out["limits"] = has_number(text, hi) and has_number(text, lo)
            out.pop("nominal", None)
        else:
            dev = [v for v in (s.get("upper"), s.get("lower")) if v]
            if dev:
                out["deviations"] = all(has_number(text, abs(v)) for v in dev)
    if k == "finish":
        out["Ra"] = norm(c["text"].split()[-1]) in norm(text)
    if k == "note":
        out["text"] = norm(c["text"]) in norm(text)
    if out:
        out["fully right"] = all(out.values())
    return out


def score_sheet(record, result, px_per_mm, warp=None):
    items = []
    for kind in ("gdt", "dimensions", "other"):
        for i in result[kind]:
            items.append({"kind": kind, "text": i["text"], "bbox": bbox(i["box"])})
    frames = [i for i in items if i["kind"] == "gdt"]
    texts = [i for i in items if i["kind"] != "gdt"]
    rows = []
    for c in record["callouts"]:
        if c["kind"] in ("datum", "target") or not c.get("page_frame"):
            continue
        box = [v * px_per_mm for v in c["page_frame"]]
        if warp is not None:   # the scan rotated the sheet: move the box with it
            pts = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
            box = bbox(cv2.transform(pts[None], warp)[0])
        checks = judge(c, place(box, texts), place(box, frames))
        if checks:
            kind = {"control": "frame", "thread": "thread note"}.get(c["kind"], c["kind"])
            if c["kind"] == "size" and re.search(r"THRU|↧", c.get("text") or ""):
                kind = "size hole note"
            if c["kind"] == "location" and c.get("reference"):
                kind = "reference distance"
            rows.append({"id": c["id"], "kind": kind, "checks": checks})
    return rows


def summarise(rows):
    tally = defaultdict(lambda: [0, 0])
    for r in rows:
        for check, ok in r["checks"].items():
            t = tally[f"{r['kind']}: {check}"]
            t[0] += ok
            t[1] += 1
    return {k: {"right": a, "of": b, "pct": round(100 * a / b, 1)} for k, (a, b) in sorted(tally.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tuples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="clean,scan")
    ap.add_argument("--keras-detector", action="store_true")
    args = ap.parse_args()
    from edocr2 import cli
    cli.use_tesseract()
    models = cli.load_models(None, args.keras_detector)
    summary = {}
    for variant in args.variants.split(","):
        out = Path(args.out) / variant
        out.mkdir(parents=True, exist_ok=True)
        rows, secs, failed = [], [], 0
        root = Path(args.tuples)
        for i, t in enumerate(sorted(p.parent for p in root.rglob("record.json"))):
            name = "_".join(t.relative_to(root).parts)   # ctc_01/seed101 -> ctc_01_seed101
            record = json.loads((t / "record.json").read_text(encoding="utf-8"))
            gray = render(t / "drawing.pdf")
            warp = None
            if variant == "scan":
                gray, warp = scan(gray, seed=1000 + i)
            started = time.time()
            try:
                result = cli.read_page(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), models, boxes=True)
            except Exception as e:
                failed += 1
                (out / f"{name}.error.txt").write_text(repr(e), encoding="utf-8")
                continue
            secs.append(time.time() - started)
            sheet_rows = score_sheet(record, result, gray.shape[1] / record["sheet"]["w"], warp)
            rows += sheet_rows
            (out / f"{name}.json").write_text(json.dumps({"result": result, "scores": sheet_rows}, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{variant} {name}: {secs[-1]:.1f} s", flush=True)
        summary[variant] = {"sheets": len(secs), "failed": failed,
                            "seconds_per_sheet": round(float(np.median(secs)), 1) if secs else None,
                            "scores": summarise(rows)}
    Path(args.out, "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for variant, s in summary.items():
        print(f"\n{variant}: {s['sheets']} sheets, {s['failed']} failed, {s['seconds_per_sheet']} s/sheet")
        for k, v in s["scores"].items():
            print(f"  {v['pct']:5.1f}%  {v['right']:4d}/{v['of']:<4d} {k}")


if __name__ == "__main__":
    main()
