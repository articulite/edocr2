"""Write tests/fixtures/text_layer.pdf: a made-up A3 sheet whose lettering is real PDF text, laid out
like a CAD export (border, title block, feature control frames with text in their cells,
dimensions, a hole note). Needs reportlab; the PDF is committed, so the tests do not.

    python tests/fixtures/make_text_layer_pdf.py
"""

import os

from reportlab.lib.pagesizes import A3, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

HERE = os.path.dirname(os.path.abspath(__file__))
pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Symbols", r"C:\Windows\Fonts\seguisym.ttf"))
mm = 72 / 25.4

c = canvas.Canvas(os.path.join(HERE, "text_layer.pdf"), pagesize=landscape(A3))
W, H = landscape(A3)
c.setLineWidth(0.7)
c.rect(10 * mm, 10 * mm, W - 20 * mm, H - 20 * mm)                 # border
c.rect(W - 190 * mm, 10 * mm, 180 * mm, 40 * mm)                  # title block
for y in (20, 30, 40):
    c.line(W - 190 * mm, y * mm, W - 10 * mm, y * mm)
c.setFont("Arial", 10)
c.drawString(W - 185 * mm, 43 * mm, "MATERIAL  AL 6061-T6")
c.drawString(W - 185 * mm, 33 * mm, "FINISH  ANODIZE CLEAR")
c.drawString(W - 185 * mm, 23 * mm, "SCALE  1:1")


def frame(x, y, cells):
    """A feature control frame at (x, y) mm, bottom-left; first cell in the symbol font."""
    h = 8
    for i, text in enumerate(cells):
        w = 10 if len(text) <= 1 else 6 + 2.6 * len(text)
        c.rect(x * mm, y * mm, w * mm, h * mm)
        c.setFont("Symbols" if i == 0 else "Arial", 12)
        c.drawString((x + 2) * mm, (y + 2.4) * mm, text)
        x += w


frame(60, 200, ["⌖", "Ø0.2", "A", "B", "C"])
frame(60, 180, ["⏥", "0.05"])
c.setFont("Arial", 12)
c.drawString(60 * mm, 150 * mm, "4X Ø4.55/4.3 THRU")
c.drawString(160 * mm, 150 * mm, "Ø12 H7")
c.drawString(160 * mm, 120 * mm, "25.4 ±0.1")
c.drawString(60 * mm, 120 * mm, "(58)")
c.drawString(60 * mm, 90 * mm, "BREAK ALL SHARP EDGES")
c.save()
print(os.path.join(HERE, "text_layer.pdf"))
