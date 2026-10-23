"""Measure each correction in edocr2/postfix.py on its own, on both benchmarks, from one reading.

    python benchmarks/ablate.py --drawgen DRAWGEN_EVAL_DIR [--models DIR] [--json OUT]

The pages are read once with no corrections; every combination listed is then applied to a
copy and scored, so the numbers differ only by the corrections. Reported: the drawgen
benchmark's main checks (clean and scanned) and the paper benchmark's macro recall,
precision and CER.
"""

import argparse, copy, glob, json, os, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from benchmarks import drawgen as dg, paper            # noqa: E402
from edocr2 import cli, postfix                         # noqa: E402
from edocr2.tools import train_tools                    # noqa: E402

COMBOS = [(), ("frame_decimal",), ("diameter_z",), ("six_b",), ("join_lines",), postfix.DEFAULT, postfix.ALL]
KEYS = ["frame: value", "frame: fully right", "thread note: size", "thread note: fully right",
        "size: nominal", "size hole note: fully right", "finish: Ra"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drawgen", required=True)
    ap.add_argument("--models")
    ap.add_argument("--json")
    args = ap.parse_args()
    cli.use_tesseract()
    models = cli.load_models(None, False, args.models)

    raw_dg = []   # (variant, record, result, px_per_mm, warp)
    root = Path(args.drawgen)
    for i, t in enumerate(sorted(p.parent for p in root.rglob("record.json"))):
        record = json.loads((t / "record.json").read_text(encoding="utf-8"))
        base = dg.render(t / "drawing.pdf")
        for variant in ("clean", "scan"):
            gray, warp = (base, None) if variant == "clean" else dg.scan(base, seed=1000 + i)
            r = cli.read_page(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), models, boxes=True, fixes=())
            raw_dg.append((variant, record, r, gray.shape[1] / record["sheet"]["w"], warp))
        print("read", t.name, flush=True)
    raw_paper = []
    for img_path in sorted(glob.glob(os.path.join(paper.SAMPLES, "*.jpg")) + glob.glob(os.path.join(paper.SAMPLES, "*.png"))):
        r = cli.read_page(cv2.imread(img_path), models, boxes=True, max_img_size=1240, fixes=())
        raw_paper.append((paper.labels(img_path[:-4] + ".json"), r))

    table = {}
    for combo in COMBOS:
        name = "+".join(combo) if combo else "none"
        name = {postfix.ALL: "all", postfix.DEFAULT: "default"}.get(combo, name)
        row = {}
        for variant in ("clean", "scan"):
            rows = []
            for v, record, r, k, warp in raw_dg:
                if v == variant:
                    rows += dg.score_sheet(record, postfix.apply(copy.deepcopy(r), combo), k, warp)
            s = dg.summarise(rows)
            row[variant] = {key: s[key]["pct"] for key in KEYS if key in s}
        per = []
        for gt, r in raw_paper:
            r = postfix.apply(copy.deepcopy(r), combo)
            preds = [(d["text"], np.array(d["box"])) for d in r["dimensions"]] + [("other_info", np.array(o["box"])) for o in r["other"]]
            per.append(paper.score(gt, preds))
        row["paper"] = {
            "recall": round(100 * sum(x["correct"] for x in per) / max(1, sum(x["labelled"] for x in per)), 2),
            "precision": round(100 * sum(x["correct"] for x in per) / max(1, sum(x["predicted"] for x in per)), 2),
            "CER": round(100 * train_tools.get_cer("".join(x["pred"] for x in per), "".join(x["gt"] for x in per)), 2),
        }
        table[name] = row
    print(json.dumps(table, ensure_ascii=False, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
