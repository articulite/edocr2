"""The corrections in edocr2/postfix.py: each fixes what it is for and leaves the rest alone."""

from edocr2.postfix import diameter_z, frame_decimal, join_lines, six_b


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
    assert [six_b(t) for t in ("B", "AB", "6H")] == ["B", "AB", "6H"]


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_pieces_of_one_callout_are_joined():
    items = [{"text": "29x", "box": box(100, 100, 200, 150)}, {"text": "M12", "box": box(230, 100, 330, 150)},
             {"text": "6G", "box": box(360, 100, 420, 150)},
             {"text": "4.55", "box": box(800, 90, 900, 130)}, {"text": "4.3", "box": box(800, 140, 880, 180)},
             {"text": "4x⌀", "box": box(680, 110, 780, 160)}, {"text": "THRU", "box": box(920, 110, 1040, 160)},
             {"text": "99", "box": box(2000, 2000, 2050, 2050)}, {"text": "25", "box": box(2000, 2300, 2050, 2350)}]
    assert sorted(i["text"] for i in join_lines(items)) == sorted(["29x M12 6G", "4x⌀ 4.55/4.3 THRU", "99", "25"])
