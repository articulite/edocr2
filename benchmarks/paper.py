"""The paper's benchmark: dimensions on the labelled drawings in tests/test_samples, scored as
test_all.py scores them (detection precision and recall at IoU >= 0.2, character recall and
CER on matched dimensions, micro and macro averaged), run through the current pipeline.

    python benchmarks/paper.py [--keras] [--json OUT]

test_all.py itself no longer runs: it calls ocr_dimensions without the frame argument that
was added later. The metric code here is the same; only the call is current.
"""

import argparse, glob, json, math, os, sys, time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edocr2 import cli
from edocr2.tools import train_tools

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "test_samples")


def labels(json_path):
    """(label, 4x2 points) per labelled shape, ordered as test_all.process_json_labels does."""
    out = []
    for s in json.load(open(json_path, encoding="utf-8"))["shapes"]:
        if s["shape_type"] == "rectangle":
            (x1, y1), (x2, y2) = s["points"]
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        else:
            p = s["points"]
            start = min(p, key=lambda q: (q[0], q[1]))
            pts = [start] + sorted((q for q in p if q != start), key=lambda q: math.atan2(q[1] - start[1], q[0] - start[0]))
        out.append((s["label"], np.array(pts, dtype=float).astype(int)))
    return out


def score(ground_truth, predictions, iou_thres=0.2):
    """test_all.compute_metrics, without the drawing."""
    correct, ious, total_chars, correct_chars, cum_gt, cum_pred = 0, [], 0, 0, "", ""
    for label, gbox in ground_truth:
        best, best_iou = None, 0
        for p in predictions:
            iou = train_tools.calculate_iou(p[1], gbox)
            if iou > best_iou:
                best_iou, best = iou, p
        if best_iou >= iou_thres and label != "other_info" and best[0] != "other_info":
            correct += 1
            ious.append(best_iou)
            cum_gt += label
            cum_pred += best[0]
            correct_chars += train_tools.compare_characters(label, best[0])
            total_chars += len(label)
    n_gt = sum(1 for g in ground_truth if g[0] != "other_info")
    n_pred = sum(1 for p in predictions if p[0] != "other_info")
    return {"correct": correct, "predicted": n_pred, "labelled": n_gt, "ious": ious,
            "correct_chars": correct_chars, "total_chars": total_chars, "pred": cum_pred, "gt": cum_gt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", action="store_true", help="the original TensorFlow models")
    ap.add_argument("--models", help="folder of PyTorch recognizers to use instead of edocr2/models")
    ap.add_argument("--fixes", default="all", help="corrections from edocr2/postfix.py: all, none, or a comma list")
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args()
    cli.use_tesseract()
    models = cli.load_models(None, args.keras, args.models)
    fixes = None if args.fixes == "all" else () if args.fixes == "none" else tuple(args.fixes.split(","))
    per, times = {}, []
    for img_path in sorted(glob.glob(os.path.join(SAMPLES, "*.jpg")) + glob.glob(os.path.join(SAMPLES, "*.png"))):
        name = os.path.splitext(os.path.basename(img_path))[0]
        t = time.time()
        r = cli.read_page(cv2.imread(img_path), models, boxes=True, max_img_size=1240, fixes=fixes)
        times.append(time.time() - t)
        preds = [(d["text"], np.array(d["box"])) for d in r["dimensions"]] + [("other_info", np.array(o["box"])) for o in r["other"]]
        per[name] = score(labels(img_path[:-4] + ".json"), preds)
    m = list(per.values())
    micro = {
        "precision": np.mean([x["correct"] / x["predicted"] if x["predicted"] else 0 for x in m]),
        "recall": np.mean([x["correct"] / x["labelled"] if x["labelled"] else 0 for x in m]),
        "char_recall": np.mean([100 * x["correct_chars"] / x["total_chars"] if x["total_chars"] else 0 for x in m]),
        "CER": np.mean([train_tools.get_cer(x["pred"], x["gt"]) for x in m]),
    }
    macro = {
        "precision": sum(x["correct"] for x in m) / max(1, sum(x["predicted"] for x in m)),
        "recall": sum(x["correct"] for x in m) / max(1, sum(x["labelled"] for x in m)),
        "IoU": float(np.mean([i for x in m for i in x["ious"]])),
        "char_recall": 100 * sum(x["correct_chars"] for x in m) / max(1, sum(x["total_chars"] for x in m)),
        "CER": train_tools.get_cer("".join(x["pred"] for x in m), "".join(x["gt"] for x in m)),
    }
    result = {"drawings": len(m), "seconds_per_drawing": round(float(np.mean(times)), 1),
              "micro": {k: round(float(v), 4) for k, v in micro.items()},
              "macro": {k: round(float(v), 4) for k, v in macro.items()},
              "per_drawing": {k: {"recall": round(v["correct"] / max(1, v["labelled"]), 3),
                                  "precision": round(v["correct"] / max(1, v["predicted"]), 3)} for k, v in per.items()}}
    print(json.dumps(result, indent=1))
    if args.json:
        json.dump(result, open(args.json, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
