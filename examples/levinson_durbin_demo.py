"""Demo: Levinson-Durbin -- Toeplitz solving and autoregressive modelling in O(n^2).

Solves a symmetric Toeplitz system, fits an AR(2) model to data synthesised from a known process,
forecasts several steps ahead, and draws the signal with its one-step-ahead AR predictions overlaid.

    python examples/levinson_durbin_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from levinson_durbin import (solve_toeplitz, ar_fit, ar_predict,  # noqa: E402
                             is_positive_definite)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self):
        return sum(self.u() for _ in range(12)) - 6.0


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Levinson-Durbin: O(n^2) Toeplitz solving and AR modelling\n")

    r = [4.0, 1.0, 0.0]
    b = [1.0, 2.0, 3.0]
    x = solve_toeplitz(r, b)
    print("  Toeplitz system T x = b, T built from first row [4,1,0]:")
    print(f"    b = {b}  ->  x = [{', '.join(f'{v:.4f}' for v in x)}]")
    print(f"    matrix positive definite? {is_positive_definite(r)}\n")

    # synthesise x_t = 0.75 x_{t-1} - 0.5 x_{t-2} + noise
    true_a = [0.75, -0.5]
    gen = LCG(2024)
    N = 4000
    xs = [0.0, 0.0]
    for t in range(2, N):
        xs.append(true_a[0] * xs[t - 1] + true_a[1] * xs[t - 2] + 0.3 * gen.normal())

    coeffs, reflection, err = ar_fit(xs, 2)
    print("  AR(2) fit on 4000 samples of x_t = 0.75 x_{t-1} - 0.5 x_{t-2} + noise:")
    print(f"    true coeffs      [0.75, -0.50]")
    print(f"    recovered coeffs [{coeffs[0]:.4f}, {coeffs[1]:.4f}]")
    print(f"    reflection coeffs {[round(k, 4) for k in reflection]} (all |k|<1 -> stable)")
    print(f"    prediction-error variance {err:.4f} (noise var ~ {0.3**2 * (12/12):.4f})\n")

    # forecast 5 steps ahead by feeding predictions back in
    hist = xs[:]
    fc = []
    for _ in range(5):
        nxt = ar_predict(hist, coeffs)
        fc.append(nxt)
        hist.append(nxt)
    print("  5-step forecast from the fitted model:")
    print(f"    {[round(v, 4) for v in fc]}\n")

    print("  A stationary signal has a Toeplitz covariance matrix, so its Yule-Walker normal")
    print("  equations are Toeplitz and Levinson-Durbin solves them in O(n^2) instead of O(n^3),")
    print("  yielding the AR coefficients that LPC, spectral estimation, and forecasting all use.")

    _svg(os.path.join(outdir, "levinson_durbin.svg"), xs, coeffs)
    print(f"\n  wrote {os.path.join(outdir, 'levinson_durbin.svg')}")


def _svg(path, xs, coeffs, width=760, height=380):
    # show a window of the signal with one-step-ahead predictions
    start = 200
    count = 120
    seg = xs[start:start + count]
    preds = []
    for i in range(count):
        t = start + i
        preds.append(coeffs[0] * xs[t - 1] + coeffs[1] * xs[t - 2])

    lo = min(min(seg), min(preds))
    hi = max(max(seg), max(preds))
    span = hi - lo if hi > lo else 1.0

    ox, oy = 50, 320
    pw, ph = width - 80, 250

    def px(i):
        return ox + i / (count - 1) * pw

    def py(v):
        return oy - (v - lo) / span * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'AR(2) one-step-ahead prediction of a synthesised signal</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'blue = actual signal; yellow = predicted from the two previous samples via fitted AR coeffs</text>',
    ]
    # zero line
    parts.append(f'<line x1="{ox}" y1="{py(0):.1f}" x2="{ox+pw}" y2="{py(0):.1f}" '
                 f'stroke="#30363d" stroke-width="1"/>')
    # actual
    pts = " ".join(f"{px(i):.1f},{py(seg[i]):.1f}" for i in range(count))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="1.8"/>')
    # predicted
    ptp = " ".join(f"{px(i):.1f},{py(preds[i]):.1f}" for i in range(count))
    parts.append(f'<polyline points="{ptp}" fill="none" stroke="#ffd43b" stroke-width="1.4" '
                 f'stroke-dasharray="4 2" opacity="0.9"/>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+28}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">time (samples)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
