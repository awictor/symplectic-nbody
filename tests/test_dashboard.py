"""Dashboard build smoke test: the generator runs and emits valid HTML."""

import os
import sys
import tempfile

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))


def test_dashboard_builds():
    import build_dashboard
    with tempfile.TemporaryDirectory() as d:
        argv = sys.argv
        sys.argv = ["build_dashboard", d]
        try:
            build_dashboard.main()
        finally:
            sys.argv = argv
        index = os.path.join(d, "index.html")
        assert os.path.exists(index)
        html = open(index, encoding="utf-8").read()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert html.count("<section>") == 14
        assert "<svg" in html  # at least one figure inlined
        low = html.lower()
        # the placeholder must be fully substituted
        assert "{{sections}}" not in low


if __name__ == "__main__":
    try:
        test_dashboard_builds()
        print("PASS test_dashboard_builds")
    except AssertionError as e:
        print(f"FAIL test_dashboard_builds: {e}")
        sys.exit(1)
