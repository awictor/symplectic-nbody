"""Demo: isotonic regression by pool-adjacent-violators.

Fits a monotone step function to a noisy increasing signal, shows the block structure, and applies
isotonic regression to probability calibration (turning wobbly classifier scores into monotone
probabilities). Draws the noisy data with the isotonic fit overlaid.

    python examples/isotonic_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from isotonic import isotonic_regression, blocks, IsotonicModel, sse  # noqa: E402


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state >> 8

    return nxt


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Isotonic regression (PAVA): the best monotone fit to noisy data\n")

    rng = _lcg(2024)
    # a saturating increasing curve + noise
    n = 40
    true = [3.0 / (1 + 2.718 ** (-(i - 20) * 0.3)) for i in range(n)]  # logistic ramp 0..3
    noisy = [t + (rng() / (1 << 24) - 0.5) * 1.6 for t in true]

    fit = isotonic_regression(noisy)
    bl = blocks(noisy)

    print(f"  {n} noisy samples of a monotone (logistic) curve.")
    print(f"  Isotonic fit has {len(bl)} blocks (constant runs):")
    for start, length, val in bl[:8]:
        print(f"    indices {start:>2}..{start + length - 1:<2}  value {val:6.3f}  ({length} pts)")
    if len(bl) > 8:
        print(f"    ... and {len(bl) - 8} more blocks")

    err_raw = sum((a - b) ** 2 for a, b in zip(noisy, true))
    err_fit = sum((a - b) ** 2 for a, b in zip(fit, true))
    print(f"\n  SSE to the true curve: raw noisy {err_raw:.2f} -> isotonic fit {err_fit:.2f} "
          f"({100*(1-err_fit/err_raw):.0f}% lower)")
    print(f"  Fit is monotone by construction: {all(fit[i] <= fit[i+1] + 1e-9 for i in range(n-1))}")

    # probability calibration
    print("\n  Probability calibration (isotonic): raw classifier scores -> monotone probabilities")
    scores = [0.05, 0.12, 0.2, 0.31, 0.44, 0.5, 0.62, 0.71, 0.83, 0.95]
    outcomes = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1]  # noisy binary labels
    cal = IsotonicModel(scores, outcomes)
    print(f"    {'score':>6} {'outcome':>8} {'calibrated p':>13}")
    for s, o in zip(scores, outcomes):
        print(f"    {s:>6.2f} {o:>8} {cal.predict(s):>13.3f}")
    print("    -> calibrated probabilities are monotone in the score, as a valid probability must be.")

    _svg(os.path.join(outdir, "isotonic.svg"), noisy, fit, true)
    print(f"\n  wrote {os.path.join(outdir, 'isotonic.svg')}")


def _svg(path, noisy, fit, true, width=760, height=420):
    n = len(noisy)
    lo = min(min(noisy), min(fit), min(true))
    hi = max(max(noisy), max(fit), max(true))
    x0, y0, w, h = 50, 50, width - 90, height - 110

    def sx(i):
        return x0 + w * i / (n - 1)

    def sy(v):
        return y0 + h * (1 - (v - lo) / (hi - lo or 1))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="28" fill="#e6edf3" font-size="15">'
        'Noisy data (dots), true curve (gray), isotonic fit (green step)</text>',
    ]
    # true curve
    tpts = " ".join(f"{sx(i):.1f},{sy(true[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{tpts}" fill="none" stroke="#8b949e" stroke-width="1.5" '
                 f'stroke-dasharray="4 3"/>')
    # noisy points
    for i in range(n):
        parts.append(f'<circle cx="{sx(i):.1f}" cy="{sy(noisy[i]):.1f}" r="2.5" fill="#ff922b" '
                     f'opacity="0.8"/>')
    # isotonic step function
    step = []
    for i in range(n):
        step.append(f"{sx(i):.1f},{sy(fit[i]):.1f}")
        if i + 1 < n:
            step.append(f"{sx(i+1):.1f},{sy(fit[i]):.1f}")  # horizontal within block
    parts.append(f'<polyline points="{" ".join(step)}" fill="none" stroke="#06d6a0" '
                 f'stroke-width="2.5"/>')
    parts.append(f'<text x="{x0}" y="{height-15}" fill="#8b949e" font-size="10">'
                 f'the fit never decreases; flat runs are pooled-violator blocks</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
