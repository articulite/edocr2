"""Corrections applied to what the pipeline read, each for a mistake measured on drawgen sheets
(benchmarks/drawgen.py). Each can be switched off; benchmarks/ablate.py measures them one at a
time.

frame_decimal  A GD&T tolerance read without its separator (001, 02, ⌀0005) gets it back:
               a frame value starts with 0 and is under 1, so 001 can only be 0.01. The
               recognizer drops thin points and decimal commas it was not trained on.
diameter_z     Z before a number is the diameter sign (Z12.5, 4xZ6.6): the released recognizer
               writes Z for a ⌀ it does not know.
six_b          B touching a digit, or after M, or starting a thread class, is a 6 (MB, M1B, 22.8B,
               BH): the commonest confusion in single-stroke lettering.
join_lines     Pieces of one callout read apart (the detector boxes words separately) are joined
               left to right where they stand on one line close together; pieces stacked one
               above the other at the same place (limits, deviations) are joined with '/'.
"""

import re

import numpy as np

ALL = ("frame_decimal", "diameter_z", "six_b", "join_lines")


def frame_decimal(text):
    cells = text.split("|")
    for i in range(1, len(cells)):
        m = re.fullmatch(r"(⌀?)0(\d+)([ⒺⒻⓁⓂⓅⓈⓉⓊ]*)", cells[i])
        if m:
            cells[i] = f"{m.group(1)}0.{m.group(2)}{m.group(3)}"
    return "|".join(cells)


def diameter_z(text):
    return re.sub(r"(^|[xX×\s])Z(?=\d)", r"\1⌀", text)


def six_b(text):
    text = re.sub(r"(?<=\d)B|B(?=\d)", "6", text)
    text = re.sub(r"(?<=M)B", "6", text)
    return re.sub(r"(^|[-\s])B(?=[HGhg]\b)", r"\g<1>6", text)


def _bbox(item):
    b = np.asarray(item["box"], dtype=float).reshape(-1, 2)
    return [b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()]


def _groups(n, linked):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n):
        for b in range(a + 1, n):
            if linked(a, b):
                parent[find(a)] = find(b)
    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def join_lines(items):
    """Join pieces of one callout. items: [{"text", "box"}]; returns the same shape.

    First pieces stacked one above the other at the same place become one unit, joined top to
    bottom with '/' (4.55 over 4.3 -> 4.55/4.3); then units side by side on one line, close
    together, are joined left to right with spaces (4x⌀ and 4.55/4.3 and THRU)."""
    boxes = [_bbox(i) for i in items]
    height = [b[3] - b[1] for b in boxes]

    def stacked(a, b):
        A, B, h = boxes[a], boxes[b], min(height[a], height[b])
        if h <= 0 or max(height[a], height[b]) > 2.0 * h:
            return False
        overlap_x = min(A[2], B[2]) - max(A[0], B[0])
        gap_y = max(A[1], B[1]) - min(A[3], B[3])
        return overlap_x > 0.5 * min(A[2] - A[0], B[2] - B[0]) and -0.2 * h < gap_y < 0.6 * h

    units = []
    for members in _groups(len(items), stacked):
        members.sort(key=lambda i: boxes[i][1])
        units.append({"text": "/".join(items[i]["text"] for i in members), "bbox": _union([boxes[i] for i in members]),
                      "h": min(height[i] for i in members)})

    def beside(a, b):
        A, B = units[a]["bbox"], units[b]["bbox"]
        h = min(units[a]["h"], units[b]["h"])
        overlap_y = min(A[3], B[3]) - max(A[1], B[1])
        gap_x = max(A[0], B[0]) - min(A[2], B[2])
        return h > 0 and overlap_y > 0.3 * h and gap_x < 1.2 * h

    out = []
    for members in _groups(len(units), beside):
        members.sort(key=lambda u: units[u]["bbox"][0])
        x0, y0, x1, y1 = _union([units[u]["bbox"] for u in members])
        out.append({"text": " ".join(units[u]["text"] for u in members), "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]})
    return out


def apply(result, fixes=ALL):
    """Correct a read_page(..., boxes=True) result in place and return it."""
    if "frame_decimal" in fixes:
        for g in result["gdt"]:
            g["text"] = frame_decimal(g["text"])
    for key in ("dimensions", "other"):
        for d in result[key]:
            if "diameter_z" in fixes:
                d["text"] = diameter_z(d["text"])
            if "six_b" in fixes:
                d["text"] = six_b(d["text"])
    if "join_lines" in fixes:
        result["dimensions"] = join_lines(result["dimensions"])
    return result
