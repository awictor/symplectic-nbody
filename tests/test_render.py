"""SVG renderer tests: valid output, correct structure, no NaN/Inf leakage."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from systems import figure_eight  # noqa: E402
from render_svg import render, _project  # noqa: E402


def test_projection_planes():
    p = (1.0, 2.0, 3.0)
    assert _project(p, "xy") == (1.0, 2.0)
    assert _project(p, "xz") == (1.0, 3.0)
    assert _project(p, "yz") == (2.0, 3.0)


def test_render_writes_valid_svg():
    sys_ = figure_eight()
    traj = sys_.record("forest_ruth", 0.005, 1000, sample_every=5)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.svg")
        svg = render(traj, path, title="test")
        assert os.path.exists(path)
        assert svg.startswith("<svg")
        assert svg.rstrip().endswith("</svg>")
        # one polyline per body
        assert svg.count("<polyline") == len(traj)
        # no numerical garbage
        low = svg.lower()
        assert "nan" not in low and "inf" not in low


def test_title_is_escaped():
    traj = [[(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
    with tempfile.TemporaryDirectory() as d:
        svg = render(traj, os.path.join(d, "t.svg"), title="a<b>&c")
        assert "&lt;b&gt;" in svg and "&amp;c" in svg
        assert "<b>" not in svg


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
