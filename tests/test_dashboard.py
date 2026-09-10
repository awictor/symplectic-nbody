"""Dashboard build smoke test: the generator runs and emits valid HTML."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))


OUTDIR = os.path.join(HERE, "..", "examples", "output")


def test_dashboard_builds():
    # Rebuild in --fast mode against the committed output dir: reuses the cached
    # demo text and pre-rendered SVGs so the smoke test is instant. It checks the
    # generator's HTML assembly (structure, substitution), not the physics --
    # each demo's own test suite covers correctness.
    import build_dashboard
    argv = sys.argv
    sys.argv = ["build_dashboard", OUTDIR, "--fast"]
    try:
        build_dashboard.main()
    finally:
        sys.argv = argv
    index = os.path.join(OUTDIR, "index.html")
    assert os.path.exists(index)
    html = open(index, encoding="utf-8").read()
    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html
    assert html.count("<section>") == 111
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
