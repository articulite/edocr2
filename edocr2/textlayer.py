"""Reading a PDF page's own text instead of recognising it.

A drawing exported from CAD (NX, SolidWorks, ...) keeps its lettering as text with positions,
symbols included when the symbol font maps them to Unicode (NX's NXANSISymbols does). Reading
that is exact where OCR guesses. The page is still rendered for segmentation, which finds the
GD&T frames and tables; the text that falls in each is taken from the PDF.
"""

import re

import numpy as np

# fewer characters than this and the page is treated as an image (a scan, or text as lines)
MIN_CHARS = 40


def text_runs(page, scale):
    """Each run of text on a pypdfium2 page as {"text", "box"}, the box in pixels of the page
    rendered at `scale` pixels per point, corners clockwise from top-left."""
    textpage = page.get_textpage()
    height = page.get_height()
    runs = []
    for i in range(textpage.count_rects()):
        left, bottom, right, top = textpage.get_rect(i)
        text = textpage.get_text_bounded(left, bottom, right, top).strip()
        if not text:
            continue
        x0, x1 = left * scale, right * scale
        y0, y1 = (height - top) * scale, (height - bottom) * scale
        runs.append({"text": text, "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]})
    return runs


def has_text(page):
    return len(page.get_textpage().get_text_range().strip()) >= MIN_CHARS


def _centre(run):
    b = np.asarray(run["box"])
    return b[:, 0].mean(), b[:, 1].mean()


def _inside(run, rect, margin=2):
    x, y = _centre(run)
    return rect.x - margin <= x <= rect.x + rect.w + margin and rect.y - margin <= y <= rect.y + rect.h + margin


# a dimension: has a digit and is short; anything else on the sheet is "other" text
DIMENSION = re.compile(r"^(?=.*\d).{1,40}$")


def read_text_page(img, runs, border=None):
    """The same result as cli.read_page(..., boxes=True), from the page's text runs. border: as
    read_page's: with "ring", GD&T frames anywhere on the page count."""
    import os
    from edocr2 import tools
    border = border or os.environ.get("EDOCR2_BORDER", "inner")
    _, frame, gdt_boxes, tables, _ = tools.layer_segm.segment_img(img, autoframe=True, frame_thres=0.7, GDT_thres=0.02, binary_thres=127, border="page" if border == "ring" else "inner")
    used = set()
    gdt = []
    for block in gdt_boxes:
        for cluster, cells in block.items():
            # cells in reading order, each cell's runs left to right; a frame needs a number
            cells = tools.ocr_pipelines.sort_gdt_boxes(list(cells), 3)
            texts = []
            for c in cells:
                inside = [i for i, r in enumerate(runs) if i not in used and _inside(r, c)]
                used.update(inside)
                texts.append(" ".join(runs[i]["text"] for i in sorted(inside, key=lambda i: _centre(runs[i])[0])))
            text = "|".join(t for t in texts if t)
            if any(ch.isdigit() for ch in text):
                x0, y0 = cluster.x, cluster.y
                gdt.append({"text": text, "box": [[x0, y0], [x0 + cluster.w, y0], [x0 + cluster.w, y0 + cluster.h], [x0, y0 + cluster.h]]})
    table_texts = []
    for block in tables:
        for cluster, cells in block.items():
            inside = [i for i, r in enumerate(runs) if i not in used and _inside(r, cluster)]
            used.update(inside)
            if inside:
                table_texts.append(" ".join(runs[i]["text"] for i in sorted(inside, key=lambda i: (_centre(runs[i])[1], _centre(runs[i])[0]))))
    dims, other = [], []
    for i, r in enumerate(runs):
        if i in used:
            continue
        (dims if DIMENSION.match(r["text"]) else other).append(r)
    return {"gdt": gdt, "dimensions": dims, "other": other, "tables": table_texts}
