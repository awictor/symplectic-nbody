"""Demo: Savitzky-Golay smoothing that preserves peaks, and noisy-data differentiation.

Smooths a noisy peak-laden signal, compares against a moving average (which blunts the peaks), and
computes a smoothed derivative of noisy data. Draws the noisy signal, the SG smoothing, and the
moving average together.

    python examples/savitzky_golay_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from savitzky_golay import coefficients, filter_signal  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Savitzky-Golay: smoothing that keeps peak shapes, and stable differentiation\n")

    print(f"  degree-2, window-5 smoothing coefficients: "
          f"{[round(c, 4) for c in coefficients(5, 2)]}")
    print(f"    (sum = {sum(coefficients(5,2)):.4f}, the classic -3 12 17 12 -3 / 35)")

    # a signal: two Gaussian peaks on a slope, plus noise
    state = 20260911

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    n = 200
    def clean(i):
        return (0.005 * i
                + math.exp(-((i - 60) ** 2) / (2 * 8 ** 2))
                + 0.8 * math.exp(-((i - 130) ** 2) / (2 * 5 ** 2)))
    c = [clean(i) for i in range(n)]
    noisy = [c[i] + (rng() - 0.5) * 0.25 for i in range(n)]

    sg = filter_signal(noisy, 15, 3)

    # moving average for comparison
    half = 7
    ma = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        ma.append(sum(noisy[lo:hi]) / (hi - lo))

    def mse(a, b):
        return sum((a[i] - b[i]) ** 2 for i in range(n)) / n

    print(f"\n  MSE to the clean signal:")
    print(f"    noisy input:            {mse(noisy, c):.4f}")
    print(f"    moving average:         {mse(ma, c):.4f}")
    print(f"    Savitzky-Golay:         {mse(sg, c):.4f}")
    print(f"  peak-2 height (true {max(c):.3f}): SG {max(sg):.3f}, moving avg {max(ma):.3f}")
    print(f"    -> the moving average flattens the sharp peak; SG keeps it")

    # smoothed derivative of noisy data
    d1 = filter_signal(noisy, 21, 3, deriv=1)
    # the derivative crosses zero at each peak
    peaks = [i for i in range(1, n - 1) if d1[i - 1] > 0 >= d1[i] and c[i] > 0.5]
    print(f"\n  Smoothed 1st derivative finds peak locations (zero crossings): {peaks}")
    print(f"    (true peaks near 60 and 130 -- differentiating raw noisy data would be hopeless)")

    print("\n  Savitzky-Golay fits a low-degree polynomial to a sliding window by least squares and")
    print("  takes the centre value -- a fixed convolution. The polynomial follows a peak's")
    print("  curvature, so unlike a moving average it smooths without blunting features, and its")
    print("  derivative gives a stable estimate of the signal's slope.")

    _svg(os.path.join(outdir, "savitzky_golay.svg"), noisy, sg, ma, c)
    print(f"\n  wrote {os.path.join(outdir, 'savitzky_golay.svg')}")


def _svg(path, noisy, sg, ma, clean, width=760, height=430):
    n = len(noisy)
    m_left, m_right, m_top, m_bot = 50, 40, 80, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot
    lo = min(min(noisy), min(sg), min(ma))
    hi = max(max(noisy), max(sg), max(ma))
    rng = hi - lo or 1

    def px(i):
        return m_left + i / (n - 1) * pw

    def py(v):
        return m_top + ph - (v - lo) / rng * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Savitzky-Golay vs moving average on a noisy peak signal</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'gray = noisy input, green = Savitzky-Golay (peaks kept), red = moving average (peaks blunted)</text>',
    ]

    def poly(series, color, width_px, opacity=1.0):
        pts = " ".join(f"{px(i):.1f},{py(series[i]):.1f}" for i in range(n))
        return (f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width_px}" '
                f'opacity="{opacity}"/>')

    parts.append(poly(noisy, "#8b949e", 0.8, 0.6))
    parts.append(poly(ma, "#ff6b6b", 1.8))
    parts.append(poly(sg, "#06d6a0", 2.0))

    # legend
    ly = m_top + 6
    for col, name in (("#8b949e", "noisy"), ("#06d6a0", "Savitzky-Golay"), ("#ff6b6b", "moving avg")):
        parts.append(f'<line x1="{m_left+pw-140}" y1="{ly}" x2="{m_left+pw-120}" y2="{ly}" '
                     f'stroke="{col}" stroke-width="2.5"/>')
        parts.append(f'<text x="{m_left+pw-114}" y="{ly+4}" fill="#e6edf3" font-size="10">{name}</text>')
        ly += 16

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
