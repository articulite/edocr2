"""A PDF page with its own text is read from that text, exactly. The fixture is a made-up sheet
(tests/fixtures/make_text_layer_pdf.py writes it)."""

import os

import cv2
import numpy as np
import pypdfium2

from edocr2 import textlayer

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "text_layer.pdf")


def read():
    page = pypdfium2.PdfDocument(FIXTURE)[0]
    img = cv2.cvtColor(np.array(page.render(scale=300 / 72, grayscale=True).to_pil()), cv2.COLOR_GRAY2BGR)
    r = textlayer.read_text_page(img, textlayer.text_runs(page, 300 / 72))
    return page, {k: [i["text"] if isinstance(i, dict) else i for i in v] for k, v in r.items()}


def test_a_page_with_text_is_recognised_as_such():
    page, _ = read()
    assert textlayer.has_text(page)


def test_frames_read_cell_by_cell_with_their_symbols():
    _, r = read()
    assert sorted(r["gdt"]) == ["⌖|Ø0.2|A|B|C", "⏥|0.05"]


def test_dimensions_notes_and_title_block_are_read_exactly():
    _, r = read()
    assert sorted(r["dimensions"]) == sorted(["4X Ø4.55/4.3 THRU", "Ø12 H7", "(58)", "25.4 ±0.1"])
    assert r["other"] == ["BREAK ALL SHARP EDGES"]
    assert r["tables"] == ["MATERIAL AL 6061-T6 FINISH ANODIZE CLEAR SCALE 1:1"]
