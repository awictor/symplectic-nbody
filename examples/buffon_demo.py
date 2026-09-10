"""Demo: Buffon's needle -- estimating pi by dropping sticks.

Prints the crossing probability and the pi estimate as the number of drops grows, plus the
1/sqrt(N) convergence, then draws a scatter of dropped needles across the ruled lines
(crossing ones highlighted) and the pi estimate converging toward the true value.

    python examples/buffon_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from buffon import (crossing_probability, estimate_pi, simulate,  # noqa: E402
                    monte_carlo_error, needles_for_accuracy, _Rng)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Buffon's needle: P(cross) = 2L/(pi d), so pi ~ 2 L N / (d C)\n")
    print("  Crossing probability for L = d: %.4f (= 2/pi)\n" % crossing_probability(1.0, 1.0))
    print(f"  {'drops N':>10}{'crossings':>12}{'pi estimate':>14}{'error':>12}")
    for N in (100, 1000, 10000, 100000, 1000000):
        c, pi_est = simulate(1.0, 1.0, N, seed=1)
        print(f"  {N:>10}{c:>12}{pi_est:>14.5f}{abs(pi_est-math.pi):>12.5f}")

    print("\n  Convergence is slow (Monte Carlo, error ~ 1/sqrt(N)):")
    for err in (0.05, 0.01, 0.001):
        print(f"    {err*100:.1f} % accuracy  ->  ~{needles_for_accuracy(err):,} drops")

    print("\n  pi falls out of a purely mechanical experiment -- counting how often a tossed")
    print("  stick lands across a floorboard. No measurement of pi enters anywhere; it emerges")
    print("  from the geometry of random position and angle. The first problem in geometric probability.")

    _svg(os.path.join(outdir, "buffon.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'buffon.svg')}")


def _svg(path, size=720, pad=70):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Buffon&#39;s needle</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'needles dropped on ruled lines (crossings in red); pi estimate converging (right)</text>',
    ]

    # --- left: needle scatter on ruled lines ---
    lx0, lx1 = pad, size * 0.52
    ly0, ly1 = pad + 50, size - pad
    n_lines = 6
    d_px = (lx1 - lx0) / n_lines            # line spacing on screen
    for i in range(n_lines + 1):
        gx = lx0 + i * d_px
        parts.append(f'<line x1="{gx:.1f}" y1="{ly0:.1f}" x2="{gx:.1f}" y2="{ly1:.1f}" '
                     f'stroke="#30363d" stroke-width="1.5"/>')

    # drop needles: centre x uniform, angle uniform; length = d_px (L=d)
    rng = _Rng(seed=3)
    half = d_px / 2.0
    L = d_px
    n_needles = 60
    for _ in range(n_needles):
        cx = lx0 + rng.random() * (lx1 - lx0)
        cy = ly0 + rng.random() * (ly1 - ly0 - 20) + 10
        ang = rng.random() * math.pi
        dx = (L / 2.0) * math.cos(ang)
        dy = (L / 2.0) * math.sin(ang)
        x_a, x_b = cx - dx, cx + dx
        # crosses if endpoints straddle a gridline
        def line_index(x):
            return int((x - lx0) / d_px)
        crosses = line_index(x_a) != line_index(x_b)
        col = "#ff6b6b" if crosses else "#4dabf7"
        parts.append(f'<line x1="{x_a:.1f}" y1="{cy-dy:.1f}" x2="{x_b:.1f}" y2="{cy+dy:.1f}" '
                     f'stroke="{col}" stroke-width="1.8" opacity="0.85"/>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly1+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">red = crosses a line, blue = lands between</text>')

    # --- right: pi estimate vs N (log x) ---
    rx0, rx1 = size * 0.60, size - pad
    ry0, ry1 = size - pad, pad + 60
    lnN0, lnN1 = 2.0, 6.0    # 100 .. 1e6
    pi_lo, pi_hi = 2.8, 3.5
    def NX(logN):
        return rx0 + (logN - lnN0) / (lnN1 - lnN0) * (rx1 - rx0)
    def PY(p):
        return ry0 - (p - pi_lo) / (pi_hi - pi_lo) * (ry0 - ry1)
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.4"/>')
    # true pi line
    parts.append(f'<line x1="{rx0}" y1="{PY(math.pi):.1f}" x2="{rx1}" y2="{PY(math.pi):.1f}" '
                 f'stroke="#ffd43b" stroke-width="1" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{rx1-4:.1f}" y="{PY(math.pi)-4:.1f}" fill="#ffd43b" font-size="10" '
                 f'text-anchor="end">pi</text>')
    pts = []
    for e in range(20, 61):
        logN = e / 10.0
        N = int(10 ** logN)
        _, pe = simulate(1.0, 1.0, N, seed=1)
        pe = max(pi_lo, min(pi_hi, pe))
        pts.append(f"{NX(logN):.1f},{PY(pe):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#06d6a0" stroke-width="2.2"/>')
    for e in (2, 3, 4, 5, 6):
        parts.append(f'<text x="{NX(e):.1f}" y="{ry0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">10^{e}</text>')
    for p in (3.0, 3.14, 3.4):
        parts.append(f'<text x="{rx0-4:.1f}" y="{PY(p)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{p:.2f}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+28:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">drops N (pi estimate, error ~ 1/sqrt N)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
