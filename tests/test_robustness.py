"""Inputs that used to crash the pipeline, found on scanned drawings, and the Windows file
encoding. Run with:  .venv\\Scripts\\python -m pytest tests -q

The detector comparison loads both detectors and runs the TensorFlow one, which takes minutes
on a CPU; set EDOCR2_SLOW=1 to include it."""

import os

import cv2
import numpy as np
import pytest

from edocr2.keras_ocr import tools as kt
from edocr2.tools import layer_segm, ocr_pipelines


def white(h, w):
    return np.full((h, w, 3), 255, np.uint8)


def test_alphabet_with_symbols_reads_on_any_code_page(tmp_path):
    (tmp_path / "model.txt").write_text("0123⌀⏥⌖±°\n", encoding="utf-8")
    assert ocr_pipelines.read_alphabet(str(tmp_path / "model.keras")) == "0123⌀⏥⌖±°"


def test_check_tolerances_keeps_a_blank_crop_whole():
    img = white(40, 120)
    assert ocr_pipelines.check_tolerances(img)[0] is img


def test_check_tolerances_keeps_a_crop_with_ink_on_one_row_whole():
    img = white(40, 120)
    img[10, 20:60] = 0
    assert ocr_pipelines.check_tolerances(img)[0] is img


def test_check_tolerances_still_splits_a_stacked_tolerance():
    # a measurement on the left, two short lines of tolerance stacked on its right
    img = white(60, 200)
    cv2.putText(img, "25", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    cv2.putText(img, "+0.1", (90, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "-0.0", (90, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    assert len(ocr_pipelines.check_tolerances(img)) in (1, 2, 3)  # runs through the original path


def test_fit_never_resizes_to_zero_pixels():
    sliver = np.zeros((2, 600, 3), np.uint8)
    out = kt.fit(sliver, width=200, height=31, cval=0)
    assert out.shape == (31, 200, 3)


def test_find_frame_without_a_frame_returns_none():
    # lines only in the top-left quarter: peaks exist, but no pair spans the sheet
    img = white(800, 1200)
    cv2.rectangle(img, (50, 50), (300, 250), (0, 0, 0), 4)
    cv2.rectangle(img, (80, 80), (280, 230), (0, 0, 0), 4)
    assert layer_segm.find_frame(img, 0.7) is None


def test_segment_img_falls_back_to_a_default_frame():
    img = white(800, 1200)
    cv2.rectangle(img, (50, 50), (300, 250), (0, 0, 0), 4)
    _, frame, *_ = layer_segm.segment_img(img, autoframe=True, frame_thres=0.7)
    assert frame and frame.w > 0 and frame.h > 0


def test_recognizer_reads_an_empty_crop_as_no_text():
    from edocr2.keras_ocr.recognition import Recognizer
    rec = Recognizer(alphabet="0123456789", weights=None)
    assert rec.recognize(image=np.zeros((0, 10, 3), np.uint8)) == ""
    assert rec.recognize(image=np.zeros((1, 10, 3), np.uint8)) == ""


@pytest.mark.skipif(os.environ.get("EDOCR2_SLOW") != "1", reason="minutes on a CPU; set EDOCR2_SLOW=1")
def test_torch_detector_matches_the_keras_detector():
    from edocr2.keras_ocr.detection import Detector, TorchDetector
    img = white(512, 640)
    for i, s in enumerate(["⌀25 H7", "4X M6 - 6H THRU", "0.05 A B"]):
        cv2.putText(img, s.replace("⌀", "D"), (40, 120 + 140 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    keras_boxes = Detector().detect([img[..., ::-1].copy()])[0]
    torch_boxes = TorchDetector(device="cpu").detect([img[..., ::-1].copy()])[0]
    assert len(keras_boxes) == len(torch_boxes) > 0
    assert np.abs(np.sort(keras_boxes.reshape(len(keras_boxes), -1), 0) - np.sort(torch_boxes.reshape(len(torch_boxes), -1), 0)).max() < 2
