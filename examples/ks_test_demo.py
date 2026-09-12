"""Demo: the Kolmogorov-Smirnov test -- comparing distributions by their largest CDF gap.

Runs one-sample goodness-of-fit and two-sample tests, and draws the two CDFs with the maximal vertical
gap (the KS statistic D) marked.

    python examples/ks_test_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ks_test import (ks_one_sample_test, ks_two_sample_test, empirical_cdf,  # noqa: E402
                     uniform_cdf, normal_cdf)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def uniform(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self, mu=0.0, sigma=1.0):
        return mu + sigma * (sum(self.uniform() for _ in range(12)) - 6.0)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    rng = LCG(42)

    print("Kolmogorov-Smirnov test: is the fit good? are two samples the same?\n")

    uni = [rng.uniform() for _ in range(300)]
    d, p = ks_one_sample_test(uni, uniform_cdf())
    print(f"  one-sample: 300 uniform draws vs the Uniform(0,1) CDF")
    print(f"    D = {d:.4f}, p = {p:.3f}  -> {'fit accepted' if p > 0.05 else 'fit REJECTED'} at 5%")

    d, p = ks_one_sample_test(uni, normal_cdf(0.5, 0.29))
    print(f"\n  one-sample: those uniform draws vs a Normal(0.5, 0.29) CDF")
    print(f"    D = {d:.4f}, p = {p:.4f}  -> {'fit accepted' if p > 0.05 else 'fit REJECTED'} at 5%")

    a = [rng.uniform() for _ in range(200)]
    b = [rng.uniform() for _ in range(200)]
    d, p = ks_two_sample_test(a, b)
    print(f"\n  two-sample: two uniform samples")
    print(f"    D = {d:.4f}, p = {p:.3f}  -> {'same distribution' if p > 0.05 else 'DIFFERENT'}")

    c = [rng.uniform() + 0.4 for _ in range(200)]
    d, p = ks_two_sample_test(a, c)
    print(f"\n  two-sample: uniform vs shifted-uniform (+0.4)")
    print(f"    D = {d:.4f}, p = {p:.4f}  -> {'same distribution' if p > 0.05 else 'DIFFERENT'}")

    print("\n  The statistic D is just the biggest vertical gap between the two cumulative curves --")
    print("  no assumption about their shape. Under the null its distribution is universal (the")
    print("  Kolmogorov distribution), so one critical-value table works for any continuous law.")

    _svg(os.path.join(outdir, "ks_test.svg"), a, c)
    print(f"\n  wrote {os.path.join(outdir, 'ks_test.svg')}")


def _svg(path, a, b, width=720, height=440):
    Fa = empirical_cdf(a)
    Fb = empirical_cdf(b)
    allv = sorted(set(a + b))
    lo, hi = min(allv), max(allv)
    span = hi - lo if hi > lo else 1.0

    ox, oy = 60, 380
    pw, ph = width - 100, 320

    def px(x):
        return ox + (x - lo) / span * pw

    def py(v):
        return oy - v * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'Two empirical CDFs; the KS statistic is their largest vertical gap</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'blue and green staircases; the red bar marks the maximal gap D that the test measures</text>',
    ]
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#30363d"/>')

    # sample the two CDFs on a grid, draw as polylines, and find the max gap
    N = 300
    max_gap = 0.0
    max_x = lo
    ptsa, ptsb = [], []
    for i in range(N + 1):
        x = lo + span * i / N
        va, vb = Fa(x), Fb(x)
        ptsa.append(f"{px(x):.1f},{py(va):.1f}")
        ptsb.append(f"{px(x):.1f},{py(vb):.1f}")
        if abs(va - vb) > max_gap:
            max_gap = abs(va - vb)
            max_x = x
    parts.append(f'<polyline points="{" ".join(ptsa)}" fill="none" stroke="#4dabf7" '
                 f'stroke-width="1.8"/>')
    parts.append(f'<polyline points="{" ".join(ptsb)}" fill="none" stroke="#06d6a0" '
                 f'stroke-width="1.8"/>')
    # the max-gap bar
    gx = px(max_x)
    parts.append(f'<line x1="{gx:.1f}" y1="{py(Fa(max_x)):.1f}" x2="{gx:.1f}" '
                 f'y2="{py(Fb(max_x)):.1f}" stroke="#ff6b6b" stroke-width="3"/>')
    parts.append(f'<text x="{gx+8:.0f}" y="{(py(Fa(max_x))+py(Fb(max_x)))/2:.0f}" '
                 f'fill="#ff6b6b" font-size="12">D = {max_gap:.3f}</text>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+28}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">value x -></text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
