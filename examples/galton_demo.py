"""Demo: the Galton board -- independent coin flips build a bell curve.

Prints the simulated slot histogram against the exact binomial and the shrinking
binomial-to-Gaussian distance, then draws the Monte-Carlo histogram with the CLT Gaussian
overlaid, and how the total-variation distance falls like 1/sqrt(rows).

    python examples/galton_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from galton import (slot_probabilities, normal_approx, mean_slot,  # noqa: E402
                    variance_slot, total_variation_distance, normal_pdf, simulate)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    rows = 12
    print(f"Galton board: {rows} peg rows, each bead L/R by a coin flip -> binomial -> Gaussian\n")
    sim = simulate(rows, beads=50000, seed=5)
    binom = slot_probabilities(rows)
    print(f"  {'slot':>5}{'simulated':>12}{'binomial':>11}")
    for k in range(rows + 1):
        bar = "#" * int(round(sim[k] * 120))
        print(f"  {k:>5}{sim[k]:>12.4f}{binom[k]:>11.4f}  {bar}")

    print(f"\n  mean = np = {mean_slot(rows):.1f}, variance = np(1-p) = {variance_slot(rows):.2f}")
    print("  binomial -> Gaussian distance shrinks like 1/sqrt(rows):")
    for n in (8, 16, 32, 64, 128):
        print(f"    rows = {n:>4}:  TV distance = {total_variation_distance(n):.4f}")
    print("\n  No bead is steered, yet thousands pile into a smooth bell curve -- the central")
    print("  limit theorem made physical. Bias the pegs and the whole pile slides to np.")

    _svg(os.path.join(outdir, "galton.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'galton.svg')}")


def _svg(path, w=760, h=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'The Galton board: coin flips converge to a bell curve</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'simulated slot histogram with the CLT Gaussian (left); '
        f'binomial-to-Gaussian distance ~ 1/sqrt(rows) (right)</text>',
    ]

    # left: histogram + Gaussian overlay for a fair board
    rows = 20
    sim = simulate(rows, beads=60000, seed=5)
    mu = mean_slot(rows)
    sigma = math.sqrt(variance_slot(rows))

    lx0, lx1 = 55, w // 2 - 10
    ly0, ly1 = h - 50, 62
    ymax = max(max(sim), 1.0 / (sigma * math.sqrt(2 * math.pi))) * 1.12
    bw = (lx1 - lx0) / (rows + 1)

    def LY(v):
        return ly0 - v / ymax * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    for k in range(rows + 1):
        x = lx0 + k * bw
        parts.append(f'<rect x="{x+1:.1f}" y="{LY(sim[k]):.1f}" width="{bw-2:.1f}" '
                     f'height="{ly0-LY(sim[k]):.1f}" fill="#4dabf7" opacity="0.8"/>')
    # Gaussian curve (continuous)
    gpts = []
    steps = 120
    for i in range(steps + 1):
        xk = i / steps * rows
        gx = lx0 + (xk + 0.5) * bw
        gpts.append(f"{gx:.1f},{LY(normal_pdf(xk, mu, sigma)):.1f}")
    parts.append(f'<polyline points="{" ".join(gpts)}" fill="none" stroke="#ff6b6b" stroke-width="2.5"/>')
    for k in (0, rows // 2, rows):
        parts.append(f'<text x="{lx0 + (k+0.5)*bw:.1f}" y="{ly0+15:.1f}" fill="#8b949e" '
                     f'font-size="9" text-anchor="middle">{k}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">slot ({rows} rows)</text>')
    parts.append(f'<rect x="{lx0+8}" y="{ly1}" width="9" height="9" fill="#4dabf7"/>'
                 f'<text x="{lx0+21}" y="{ly1+8}" fill="#e6edf3" font-size="9">simulated beads</text>')
    parts.append(f'<rect x="{lx0+8}" y="{ly1+14}" width="9" height="9" fill="#ff6b6b"/>'
                 f'<text x="{lx0+21}" y="{ly1+22}" fill="#e6edf3" font-size="9">CLT Gaussian</text>')

    # right: TVD vs rows (log-log-ish), with 1/sqrt(n) reference
    rx0, rx1 = w // 2 + 40, w - 30
    ry0, ry1 = h - 50, 62
    ns = [4, 8, 16, 32, 64, 128, 256]
    tvs = [total_variation_distance(n) for n in ns]
    lo, hi = math.log(ns[0]), math.log(ns[-1])
    tvmax = max(tvs) * 1.1

    def RX(n):
        return rx0 + (math.log(n) - lo) / (hi - lo) * (rx1 - rx0)

    def RY(t):
        return ry0 - t / tvmax * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    # 1/sqrt(n) reference scaled to the first point
    c = tvs[0] * math.sqrt(ns[0])
    ref = " ".join(f"{RX(n):.1f},{RY(c / math.sqrt(n)):.1f}" for n in ns)
    parts.append(f'<polyline points="{ref}" fill="none" stroke="#8b949e" stroke-width="1" '
                 f'stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{RX(ns[-1]):.1f}" y="{RY(c/math.sqrt(ns[-1]))-5:.1f}" fill="#8b949e" '
                 f'font-size="9" text-anchor="end">1/sqrt(n)</text>')
    tv = " ".join(f"{RX(n):.1f},{RY(t):.1f}" for n, t in zip(ns, tvs))
    parts.append(f'<polyline points="{tv}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    for n, t in zip(ns, tvs):
        parts.append(f'<circle cx="{RX(n):.1f}" cy="{RY(t):.1f}" r="2.5" fill="#06d6a0"/>')
        parts.append(f'<text x="{RX(n):.1f}" y="{ry0+15:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{n}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">rows n (log)</text>')
    parts.append(f'<text x="{rx0-6:.1f}" y="{ry1-2:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">TV dist</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
