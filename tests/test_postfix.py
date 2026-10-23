"""The corrections in edocr2/postfix.py: each fixes what it is for and leaves the rest alone."""

from edocr2.cli import looks_like_frame
from edocr2.postfix import apply, diameter_z, frame_decimal, has_word, join_lines, six_b


def test_frame_values_get_their_separator_back():
    assert frame_decimal("⌖|001|A|B") == "⌖|0.01|A|B"
    assert frame_decimal("⏥|02") == "⏥|0.2"
    assert frame_decimal("⌖|⌀0005Ⓜ|A") == "⌖|⌀0.005Ⓜ|A"


def test_frame_values_already_right_are_kept():
    assert frame_decimal("⌖|0.05|A") == "⌖|0.05|A"
    assert frame_decimal("∠|1|A") == "∠|1|A"
    assert frame_decimal("⌖|0,02|A") == "⌖|0,02|A"


def test_z_before_a_number_is_a_diameter():
    assert diameter_z("Z12.5") == "⌀12.5"
    assert diameter_z("4xZ6.6") == "4x⌀6.6"
    assert diameter_z("ZONE") == "ZONE"


def test_b_beside_digits_or_after_m_is_a_six():
    assert [six_b(t) for t in ("MB", "M1B", "22.8B", "BH", "M20-BG")] == ["M6", "M16", "22.86", "6H", "M20-6G"]
    assert [six_b(t) for t in ("B", "AB", "6H", "NUMBER", "4xMB")] == ["B", "AB", "6H", "NUMBER", "4xM6"]


def test_notes_and_titles_are_left_alone():
    from edocr2.postfix import apply
    r = {"gdt": [], "dimensions": [{"text": "MB", "box": box(0, 0, 1, 1)}], "other": [{"text": "MARK PART NUMBER", "box": box(0, 0, 1, 1)}], "tables": []}
    apply(r)
    assert r["dimensions"][0]["text"] == "M6" and r["other"][0]["text"] == "MARK PART NUMBER"


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_pieces_of_one_callout_are_joined():
    items = [{"text": "29x", "box": box(100, 100, 200, 150)}, {"text": "M12", "box": box(230, 100, 330, 150)},
             {"text": "6G", "box": box(360, 100, 420, 150)},
             {"text": "4.55", "box": box(800, 90, 900, 130)}, {"text": "4.3", "box": box(800, 140, 880, 180)},
             {"text": "4x⌀", "box": box(680, 110, 780, 160)}, {"text": "THRU", "box": box(920, 110, 1040, 160)},
             {"text": "99", "box": box(2000, 2000, 2050, 2050)}, {"text": "25", "box": box(2000, 2300, 2050, 2350)}]
    assert sorted(i["text"] for i in join_lines(items)) == sorted(["29x M12 6G", "4x⌀ 4.55/4.3 THRU", "99", "25"])


def test_marks_without_a_letter_or_digit_are_dropped():
    assert not has_word("++") and not has_word("") and not has_word(" ")
    assert has_word("A-A") and has_word("Ø") and has_word("0 ⌵")
    r = {"gdt": [], "dimensions": [], "other": [{"text": "++", "box": []}, {"text": "NOTES", "box": []}], "tables": []}
    assert [o["text"] for o in apply(r, ("drop_marks",))["other"]] == ["NOTES"]


def test_frames_are_told_from_blocks_read_as_frames():
    assert looks_like_frame("⌖|⌀0.025|A|B") and looks_like_frame("⏥|0.05")
    assert looks_like_frame("⌖|⌀0.2|A\n⌖|⌀0.1|A")
    # a revision block's cells, read by the GD&T recognizer
    assert not looks_like_frame("\n5|⏤|,Ⓔ⏤|AⓅ0D⏤\n0.|⏤|022026.|ⒺAⓂB0Ⓔ⏤")
    assert not looks_like_frame("A|B") and not looks_like_frame("⌖|A|B")