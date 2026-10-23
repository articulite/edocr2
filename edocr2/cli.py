"""Command line for eDOCr2: `doctor`, `setup` and `run`.

    python -m edocr2 doctor               what was found: GPU, PyTorch, TensorFlow, Tesseract, models
    python -m edocr2 setup                fetch the models and the Tesseract languages the pipeline uses
    python -m edocr2 run IN... --out DIR  read drawings (PDF or image files, or folders of them)

`run` writes one JSON per page, a summary.csv with one row per page, and prints totals only.
It never calls a language model or any network service; `setup` is the only command that
downloads, and only from the public URLs listed below.
"""

import argparse, csv, glob, json, os, re, shutil, statistics, subprocess, sys, time, traceback, urllib.request

from edocr2 import textlayer

PACKAGE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE)
MODELS = os.path.join(PACKAGE, "models")
TESSDATA = os.path.join(ROOT, "tessdata")

RELEASE = "https://github.com/javvi51/edocr2/releases/download/v1.0.0/"
MODEL_FILES = ["recognizer_gdts.keras", "recognizer_gdts.txt", "recognizer_dimensions_2.keras", "recognizer_dimensions_2.txt"]
# eng and osd ship with Tesseract; nor is used for the Ø check in Pipeline.dimension_criteria
TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata/raw/main/{}.traineddata"
LANGUAGES = ["eng", "osd", "nor"]
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
# the released dimension recognizer's alphabet: what a box may contain to be routed as a dimension
ROUTING_BASE = "0123456789AaBCDRGHhMmnxZt(),.+-±:/°\"⌀="


# ───────────────────────── finding things ─────────────────────────

def find_tesseract():
    """The Tesseract executable: $TESSERACT_CMD, then PATH, then the usual Windows install places."""
    candidates = [os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"),
                  r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                  os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe")]
    return next((c for c in candidates if c and os.path.isfile(c)), None)


def system_tessdata(tesseract):
    return os.path.join(os.path.dirname(tesseract), "tessdata") if tesseract else None


def use_tesseract():
    """Point pytesseract at Tesseract and at this repository's tessdata, which holds every language used."""
    import pytesseract
    cmd = find_tesseract()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    if all(os.path.isfile(os.path.join(TESSDATA, f"{l}.traineddata")) for l in LANGUAGES):
        os.environ["TESSDATA_PREFIX"] = TESSDATA
    return cmd


TORCH_MODELS = ["recognizer_gdts.pt", "recognizer_dimensions_2.pt"]


def missing_models():
    return [f for f in MODEL_FILES + TORCH_MODELS if not os.path.isfile(os.path.join(MODELS, f))]


def missing_languages():
    return [l for l in LANGUAGES if not os.path.isfile(os.path.join(TESSDATA, f"{l}.traineddata"))]


# ───────────────────────── doctor ─────────────────────────

def doctor(args):
    ok = True
    def line(name, good, detail):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'ok' if good else '!!'}] {name:12s} {detail}")

    print("eDOCr2 doctor")
    line("python", sys.version_info[:2] >= (3, 9), f"{sys.version.split()[0]} at {sys.executable}")
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        line("pytorch", True, f"{torch.__version__}, " + (f"GPU: {gpu}" if gpu else "no GPU, the detector will run on the CPU"))
    except ImportError:
        line("pytorch", False, "not installed; run setup")
    # TensorFlow is only for --keras and for converting the released weights once
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        line("tensorflow", True, f"{tf.__version__} (only for --keras and converting weights)")
    except ImportError:
        line("tensorflow", True, "not installed: fine unless you use --keras")
    cmd = find_tesseract()
    if cmd:
        version = subprocess.run([cmd, "--version"], capture_output=True, text=True).stdout.split("\n")[0]
        line("tesseract", True, f"{version} at {cmd}")
    else:
        line("tesseract", False, "not found; install it (Windows: winget install --id UB-Mannheim.TesseractOCR) or set TESSERACT_CMD")
    langs = missing_languages()
    line("languages", not langs, f"{', '.join(LANGUAGES)} in {TESSDATA}" if not langs else f"missing {', '.join(langs)}; run setup")
    models = missing_models()
    line("models", not models, MODELS if not models else f"missing {', '.join(models)}; run setup")
    print("ready" if ok else "not ready: fix the lines marked !!")
    return 0 if ok else 1


# ───────────────────────── setup ─────────────────────────

def download(url, path):
    print(f"  downloading {url}")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)


def setup(args):
    os.makedirs(MODELS, exist_ok=True)
    for f in missing_models():
        if f in MODEL_FILES:
            download(RELEASE + f, os.path.join(MODELS, f))
    # the PyTorch recognizers are the released Keras weights, converted once (with TensorFlow)
    for f in TORCH_MODELS:
        if not os.path.isfile(os.path.join(MODELS, f)):
            from edocr2.torch_recognizer import convert
            print(f"  converting {f}")
            convert(os.path.join(MODELS, f[:-3] + ".keras"))
    os.makedirs(TESSDATA, exist_ok=True)
    system = system_tessdata(find_tesseract())
    for lang in missing_languages():
        local = os.path.join(system, f"{lang}.traineddata") if system else None
        if local and os.path.isfile(local):
            shutil.copyfile(local, os.path.join(TESSDATA, f"{lang}.traineddata"))
            print(f"  copied {lang} from {system}")
        else:
            download(TESSDATA_URL.format(lang), os.path.join(TESSDATA, f"{lang}.traineddata"))
    # the detector's weights are fetched and checked on first use; do it now so `run` works offline
    from edocr2.keras_ocr import craft, tools
    cfg = craft.PRETRAINED_WEIGHTS[("clovaai_general", True)]
    tools.download_and_verify(url=cfg["url"], filename=cfg["filename"], sha256=cfg["sha256"])
    print("setup done")
    return doctor(args)


# ───────────────────────── run ─────────────────────────

def pages(inputs, dpi, use_text=True):
    """(name, BGR image, text runs or None) for every page of every PDF and every image file
    under the inputs. A PDF page with its own text gives its runs, unless use_text is False."""
    import cv2, numpy as np
    files = []
    for i in inputs:
        if os.path.isdir(i):
            files += sorted(p for p in glob.glob(os.path.join(i, "**", "*"), recursive=True)
                            if p.lower().endswith(IMAGE_TYPES + (".pdf",)))
        else:
            files.append(i)
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        if path.lower().endswith(".pdf"):
            import pypdfium2
            doc = pypdfium2.PdfDocument(path)
            for n, page in enumerate(doc):
                gray = np.array(page.render(scale=dpi / 72, grayscale=True).to_pil())
                runs = textlayer.text_runs(page, dpi / 72) if use_text and textlayer.has_text(page) else None
                yield (stem if len(doc) == 1 else f"{stem}_p{n + 1}"), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), runs
        else:
            yield stem, cv2.imread(path), None


def load_models(device, keras=False, folder=None):
    """(GD&T recognizer, dimension recognizer, its alphabet, detector). PyTorch unless keras is
    set, which gives the original TensorFlow models; `folder` holds other PyTorch recognizers
    (recognizer_gdts.pt, recognizer_dimensions_2.pt)."""
    if not keras:
        from edocr2.keras_ocr.craft import TorchDetector
        from edocr2.torch_recognizer import TorchRecognizer
        folder = folder or MODELS
        recognizer_gdt = TorchRecognizer.load(os.path.join(folder, "recognizer_gdts.pt"), device)
        recognizer_dim = TorchRecognizer.load(os.path.join(folder, "recognizer_dimensions_2.pt"), device)
        # The third item is the alphabet the pipeline routes boxes by, not the recognizer's:
        # a widened recognizer would otherwise pull title blocks and notes in as dimensions.
        routing = recognizer_dim.alphabet
        if len(routing) > len(ROUTING_BASE):
            routing = ROUTING_BASE + "XTUIN↧⌴⌵□×"
        return recognizer_gdt, recognizer_dim, routing, TorchDetector(device=device)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)   # before the import, which already warns
    from edocr2 import tools
    from edocr2.keras_ocr.recognition import Recognizer
    from edocr2.keras_ocr.detection import Detector
    gdt_model, dim_model = os.path.join(MODELS, "recognizer_gdts.keras"), os.path.join(MODELS, "recognizer_dimensions_2.keras")
    # weights=None: every weight comes from the .keras file, so keras-ocr's backbone is not fetched
    recognizer_gdt = Recognizer(alphabet=tools.ocr_pipelines.read_alphabet(gdt_model), weights=None)
    recognizer_gdt.model.load_weights(gdt_model)
    alphabet_dim = tools.ocr_pipelines.read_alphabet(dim_model)
    recognizer_dim = Recognizer(alphabet=alphabet_dim, weights=None)
    recognizer_dim.model.load_weights(dim_model)
    return recognizer_gdt, recognizer_dim, alphabet_dim, Detector()


def read_page(img, models, language="eng", boxes=False, max_img_size=1048, fixes=None):
    """The steps of test_drawing.py, without the language-model ones, then the corrections in
    edocr2/postfix.py (its DEFAULT unless `fixes` names which; () for none). With boxes=True
    each item is {"text", "box"}, the box as four corner points in page pixels."""
    from edocr2 import postfix
    r = _read_page(img, models, language, max_img_size)
    postfix.apply(r, postfix.DEFAULT if fixes is None else fixes)
    if boxes:
        return r
    return {k: [i["text"] for i in v] if k != "tables" else v for k, v in r.items()}


def _read_page(img, models, language, max_img_size):
    import numpy as np
    from edocr2 import tools
    recognizer_gdt, recognizer_dim, alphabet_dim, detector = models
    _, frame, gdt_boxes, tables, dim_boxes = tools.layer_segm.segment_img(img, autoframe=True, frame_thres=0.7, GDT_thres=0.02, binary_thres=127)
    work = img.copy()
    table_results, _, work = tools.ocr_pipelines.ocr_tables(tables, work, language)
    gdt_results, gdt_blocks, work = tools.ocr_pipelines.ocr_gdt(work, gdt_boxes, recognizer_gdt)
    if frame:
        work = work[frame.y: frame.y + frame.h, frame.x: frame.x + frame.w]
    dimensions, other_info, _, _ = tools.ocr_pipelines.ocr_dimensions(
        work, detector, recognizer_dim, alphabet_dim, frame, dim_boxes, cluster_thres=20, max_img_size=max_img_size, language=language, backg_save=False)
    # dimensions are found inside the drawing frame; put their boxes back on the page
    dx, dy = (frame.x, frame.y) if frame else (0, 0)
    on_page = lambda b: (np.asarray(b, dtype=float) + [dx, dy]).tolist()
    def block_box(block):
        cells = [c for v in block.values() for c in v]
        x0, y0 = min(c.x for c in cells), min(c.y for c in cells)
        x1, y1 = max(c.x + c.w for c in cells), max(c.y + c.h for c in cells)
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return {"gdt": [{"text": g[0], "box": block_box(b)} for g, b in zip(gdt_results, gdt_blocks)],
            "dimensions": [{"text": d[0], "box": on_page(d[1])} for d in dimensions],
            "other": [{"text": o[0], "box": on_page(o[1])} for o in other_info],
            "tables": [" ".join(t["text"] for t in table) for table in table_results]}


def number(s):
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def page_summary(r):
    """Counts of what drives machining cost, from what was read. Rough by design: a misread
    callout moves a count by one."""
    texts = r["dimensions"] + r["other"] + r["gdt"]
    blob = " ".join(texts)
    frame_values = [number(m) for g in r["gdt"] for m in re.findall(r"(?:^|\|)⌀?(\d*[.,]\d+)", g)]
    frame_values = [v for v in frame_values if v]
    pm = [2 * number(m) for m in re.findall(r"±\s*(\d*[.,]\d+)", blob) if number(m)]
    # stacked or bracketed deviations: +0.025/0, +0.02 -0.01
    pm += [(number(a) or 0) + (number(b) or 0) for a, b in re.findall(r"\+\s*(\d*[.,]?\d+)\s*/?\s*-?\s*(\d*[.,]?\d+)", blob)
           if (number(a) or 0) + (number(b) or 0) > 0]
    # limit dimensions read as one callout: Ø5.42 Ø5.31
    for d in r["dimensions"]:
        nums = [number(n) for n in re.findall(r"\d+[.,]\d+", d)]
        if len(nums) == 2 and None not in nums and 0 < abs(nums[0] - nums[1]) < 0.5:
            pm.append(round(abs(nums[0] - nums[1]), 4))
    ra = [number(m) for m in re.findall(r"Ra\s*(\d+(?:[.,]\d+)?)", blob) if number(m)]
    return {
        "frames": len(r["gdt"]),
        "tightest_frame_tolerance": min(frame_values) if frame_values else None,
        "dimensions": len(r["dimensions"]),
        "toleranced_dimensions": len(pm),
        "tightest_tolerance_band": min(pm) if pm else None,
        "fit_classes": len(re.findall(r"(?<![A-Za-z])(?:H|G|F|JS|K|P|js|h|g|f|k|n|p|r|s)[5-9](?!\d)", blob)),
        "threads": len(re.findall(r"(?<![A-Za-z])M\d+(?:[.,]\d+)?(?:\s*[xX×]\s*\d+(?:[.,]\d+)?)?", blob)),
        "finest_ra": min(ra) if ra else None,
        "other_text": len(r["other"]),
    }


def run(args):
    import cv2
    os.makedirs(args.out, exist_ok=True)
    if missing_models():
        sys.exit("models missing: run setup first")
    if not use_tesseract():
        sys.exit("Tesseract not found: run doctor")
    if missing_languages():
        sys.exit(f"Tesseract languages missing ({', '.join(missing_languages())}): run setup")
    models = None   # loaded at the first page that needs recognising

    rows, failed, seconds, from_text = [], 0, [], 0
    for name, img, runs in pages(args.inputs, args.dpi, use_text=not args.ocr):
        out_json = os.path.join(args.out, name + ".json")
        if args.skip_done and os.path.exists(out_json):
            continue
        t = time.time()
        try:
            if img is None:
                raise ValueError("could not read the image")
            if runs is not None:
                full = textlayer.read_text_page(img, runs)
                r = {k: [i["text"] for i in v] if k != "tables" else v for k, v in full.items()}
                r["source"] = "text layer"
                from_text += 1
            else:
                if models is None:
                    t0 = time.time()
                    models = load_models("cpu" if args.cpu else None, args.keras, args.models)
                    print(f"models loaded in {time.time() - t0:.0f} s", flush=True)
                    t = time.time()
                from edocr2 import postfix
                fixes = () if args.no_fixes else postfix.DEFAULT + (("join_lines",) if args.join_lines else ())
                r = read_page(img, models, fixes=fixes)
                r["source"] = "ocr"
            r["seconds"] = round(time.time() - t, 1)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=1)
            rows.append({"page": name, "seconds": r["seconds"], **page_summary(r)})
            seconds.append(r["seconds"])
        except Exception:
            failed += 1
            with open(os.path.join(args.out, name + ".error.txt"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        done = len(rows) + failed
        if done % 5 == 0:
            print(f"  {done} pages", flush=True)

    if rows:
        with open(os.path.join(args.out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # totals only: no page names and no text from the drawings
    def count(key, test):
        return sum(1 for r in rows if r[key] is not None and test(r[key]))
    totals = {
        "pagesRead": len(rows), "pagesFailed": failed, "pagesReadFromTheirTextLayer": from_text,
        "medianSecondsPerPage": statistics.median(seconds) if seconds else None,
        "pagesWith": {
            "GD&T frames": count("frames", lambda v: v > 0),
            "toleranced dimensions": count("toleranced_dimensions", lambda v: v > 0),
            "fit classes": count("fit_classes", lambda v: v > 0),
            "threads": count("threads", lambda v: v > 0),
            "an Ra callout": count("finest_ra", lambda v: True),
        },
        "tightestFrameTolerance": {f"<= {b}": count("tightest_frame_tolerance", lambda v, b=b: v <= b) for b in (0.01, 0.025, 0.05, 0.1)},
        "tightestDimensionTolerance": {f"<= {b}": count("tightest_tolerance_band", lambda v, b=b: v <= b) for b in (0.01, 0.025, 0.05, 0.1)},
        "totals": {k: sum(r[k] for r in rows) for k in ("frames", "dimensions", "toleranced_dimensions", "fit_classes", "threads")},
        "results": os.path.abspath(args.out),
    }
    print(json.dumps(totals, ensure_ascii=False, indent=1))
    return 0 if not failed else 2


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="edocr2", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check what is installed")
    sub.add_parser("setup", help="fetch models and Tesseract languages")
    r = sub.add_parser("run", help="read drawings")
    r.add_argument("inputs", nargs="+", help="PDF or image files, or folders of them")
    r.add_argument("--out", required=True, help="folder for the results")
    r.add_argument("--dpi", type=int, default=300, help="resolution PDFs are rendered at (default 300)")
    r.add_argument("--cpu", action="store_true", help="run the detector on the CPU even when there is a GPU")
    r.add_argument("--keras", action="store_true", help="use the original TensorFlow detector and recognizers (slow)")
    r.add_argument("--skip-done", action="store_true", help="skip pages already in --out")
    r.add_argument("--ocr", action="store_true", help="recognise every page, even PDF pages that carry their own text")
    r.add_argument("--no-fixes", action="store_true", help="leave out the corrections in edocr2/postfix.py")
    r.add_argument("--join-lines", action="store_true", help="also join the words of one callout read apart (4x M12 6G)")
    r.add_argument("--models", help="folder with recognizer_gdts.pt and recognizer_dimensions_2.pt to use instead of edocr2/models")
    args = p.parse_args(argv)
    return {"doctor": doctor, "setup": setup, "run": run}[args.command](args)
