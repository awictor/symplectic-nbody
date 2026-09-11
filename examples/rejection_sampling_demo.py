"""Demo: rejection sampling -- darts under a curve.

Samples a bimodal density by throwing darts under a box envelope and keeping those below the
curve, reports the acceptance rate, and confirms the kept samples match the target. Draws the
accepted (green) vs rejected (red) darts under the density, and the resulting histogram.

    python examples/rejection_sampling_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rejection_sampling import (sample_interval, theoretical_acceptance,  # noqa: E402
                                mean, variance, chi_square_fit, histogram,
                                _estimate_max, _integrate, _Rng)


# a bimodal target: two Gaussian bumps -- hard to sample directly, easy to evaluate
def target(x):
    return math.exp(-0.5 * ((x + 1.5) / 0.5) ** 2) + 0.7 * math.exp(-0.5 * ((x - 1.5) / 0.7) ** 2)


A, B = -4.0, 4.0


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Rejection sampling: keep darts thrown under the envelope that land below f(x)\n")
    samples, acc = sample_interval(target, A, B, 100000, seed=1)
    print(f"  target: a bimodal density on [{A}, {B}] (two unequal Gaussian bumps)")
    print(f"  drew 100000 samples; acceptance rate {acc:.3f} "
          f"(theory {theoretical_acceptance(target, A, B):.3f})")
    print(f"  sample mean {mean(samples):+.4f}, variance {variance(samples):.4f}")
    print(f"  histogram chi-square vs target (12 bins): {chi_square_fit(samples, target, A, B, 12):.2f}\n")

    print("  Acceptance = area under f / area of the box, so a tighter envelope is faster:")
    print(f"  {'envelope M':>12}{'acceptance':>12}")
    for m in (_estimate_max(target, A, B) * 1.01, 2.0, 4.0):
        print(f"  {m:>12.2f}{theoretical_acceptance(target, A, B, m):>12.3f}")
    print("\n  It samples ANY density you can evaluate -- even unnormalized ones -- which is why")
    print("  it underlies Bayesian computation. The cost is the wasted darts: a loose envelope")
    print("  or a high dimension makes acceptance tiny, which is what MCMC methods work around.")

    _svg(os.path.join(outdir, "rejection_sampling.svg"), samples)
    print(f"\n  wrote {os.path.join(outdir, 'rejection_sampling.svg')}")


def _svg(path, samples, w=760, h=430):
    m = _estimate_max(target, A, B) * 1.01

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Rejection sampling: darts under a curve</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'green darts (below f) are kept, red (above f) rejected; '
        f'the kept x-values form the target histogram</text>',
    ]

    # top: dart scatter under the envelope
    tx0, tx1 = 50, w - 30
    ty0, ty1 = 210, 62

    def TX(x):
        return tx0 + (x - A) / (B - A) * (tx1 - tx0)

    def TY(y):
        return ty0 - y / m * (ty0 - ty1)

    # envelope box (M) and the target curve
    parts.append(f'<rect x="{TX(A):.1f}" y="{TY(m):.1f}" width="{TX(B)-TX(A):.1f}" '
                 f'height="{ty0-TY(m):.1f}" fill="none" stroke="#30363d" stroke-width="1"/>')
    curve = []
    steps = 240
    for i in range(steps + 1):
        x = A + (B - A) * i / steps
        curve.append(f"{TX(x):.1f},{TY(target(x)):.1f}")
    parts.append(f'<polyline points="{" ".join(curve)}" fill="none" stroke="#ffd43b" stroke-width="2"/>')
    # throw a modest number of visible darts
    rng = _Rng(seed=99)
    for _ in range(1400):
        x = rng.uniform(A, B)
        y = rng.random() * m
        col = "#06d6a0" if y <= target(x) else "#ff6b6b"
        parts.append(f'<circle cx="{TX(x):.1f}" cy="{TY(y):.1f}" r="1.3" fill="{col}" opacity="0.7"/>')
    parts.append(f'<text x="{TX(B):.1f}" y="{TY(m)-4:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">envelope M</text>')

    # bottom: histogram of the accepted samples vs target
    bins = 60
    hist = histogram(samples, A, B, bins)
    hx0, hx1 = 50, w - 30
    hy0, hy1 = h - 45, 250
    hmax = max(hist) * 1.1

    def HX(i):
        return hx0 + i / bins * (hx1 - hx0)

    def HY(c):
        return hy0 - c / hmax * (hy0 - hy1)

    bw = (hx1 - hx0) / bins
    for i, c in enumerate(hist):
        parts.append(f'<rect x="{HX(i):.1f}" y="{HY(c):.1f}" width="{max(bw-0.4,0.4):.1f}" '
                     f'height="{hy0-HY(c):.1f}" fill="#4dabf7" opacity="0.8"/>')
    # overlay target scaled to the histogram
    n = len(samples)
    width = (B - A) / bins
    norm = _integrate(target, A, B)
    tcurve = []
    for i in range(steps + 1):
        x = A + (B - A) * i / steps
        expected = n * target(x) / norm * width
        tcurve.append(f"{tx0 + (x-A)/(B-A)*(hx1-hx0):.1f},{HY(expected):.1f}")
    parts.append(f'<polyline points="{" ".join(tcurve)}" fill="none" stroke="#ffd43b" stroke-width="2.5"/>')
    parts.append(f'<text x="{(hx0+hx1)/2:.1f}" y="{hy0+14:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">accepted-sample histogram (bars) vs target density (curve)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
