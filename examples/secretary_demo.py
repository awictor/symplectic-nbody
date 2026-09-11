"""Demo: the secretary problem -- optimal stopping and the 1/e rule.

Prints the optimal cutoff and win probability against a Monte-Carlo run for several n, then
draws the win-probability-versus-cutoff curve (peaking near n/e) and the convergence of the
optimal look-fraction and win probability to 1/e as n grows.

    python examples/secretary_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from secretary import (win_probability, optimal_cutoff, optimal_fraction,  # noqa: E402
                       simulate)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    inv_e = 1.0 / math.e
    print("Secretary problem: reject the first ~37%, then take the next record. Win ~37%.\n")
    print(f"  {'n':>6}{'r* (cutoff)':>13}{'look frac':>11}{'P(win)':>9}{'sim':>9}")
    for n in (10, 50, 100, 500, 1000):
        r, p = optimal_cutoff(n)
        sim = simulate(n, r, trials=6000, seed=5)
        print(f"  {n:>6}{r:>13}{optimal_fraction(n):>11.4f}{p:>9.4f}{sim:>9.4f}")

    print(f"\n  Both the optimal look-fraction and the win probability tend to 1/e = {inv_e:.4f}")
    print("  as n grows: look at (and reject) 37% of the candidates, then leap at the next one")
    print("  better than all of them, and you land the very best about 37% of the time -- no")
    print("  matter how large n is. The same rule governs flat-hunting, parking, and auctions.")

    _svg(os.path.join(outdir, "secretary.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'secretary.svg')}")


def _svg(path, w=760, h=380):
    inv_e = 1.0 / math.e

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'The secretary problem: the 1/e optimal-stopping rule</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'win probability vs cutoff for n=100 (left); optimum converging to 1/e (right)</text>',
    ]

    # left: P(win) vs cutoff fraction for n = 100
    n = 100
    lx0, lx1 = 55, w // 2 - 20
    ly0, ly1 = h - 50, 60
    pmax = 0.45

    def LX(frac):
        return lx0 + frac * (lx1 - lx0)

    def LY(p):
        return ly0 - p / pmax * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    pts = [f"{LX((r-1)/n):.1f},{LY(win_probability(n, r)):.1f}" for r in range(1, n + 1)]
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="2.5"/>')
    # mark the 1/e look fraction and the 1/e probability
    parts.append(f'<line x1="{LX(inv_e):.1f}" y1="{ly1}" x2="{LX(inv_e):.1f}" y2="{ly0}" '
                 f'stroke="#ff922b" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{LX(inv_e)+4:.1f}" y="{ly1+12:.1f}" fill="#ff922b" font-size="9">'
                 f'look 1/e</text>')
    parts.append(f'<line x1="{lx0}" y1="{LY(inv_e):.1f}" x2="{lx1}" y2="{LY(inv_e):.1f}" '
                 f'stroke="#06d6a0" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{lx1-4:.1f}" y="{LY(inv_e)-4:.1f}" fill="#06d6a0" font-size="9" '
                 f'text-anchor="end">P = 1/e</text>')
    for frac in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{LX(frac):.1f}" y="{ly0+16:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{frac:.1f}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">look fraction (r-1)/n</text>')

    # right: optimal fraction and probability vs n (log-spaced)
    rx0, rx1 = w // 2 + 45, w - 30
    ry0, ry1 = h - 50, 60
    ns = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
    lo, hi = math.log(ns[0]), math.log(ns[-1])

    def RX(nn):
        return rx0 + (math.log(nn) - lo) / (hi - lo) * (rx1 - rx0)

    def RY(v):
        return ry0 - (v - 0.25) / (0.55 - 0.25) * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    # 1/e reference line
    parts.append(f'<line x1="{rx0}" y1="{RY(inv_e):.1f}" x2="{rx1}" y2="{RY(inv_e):.1f}" '
                 f'stroke="#8b949e" stroke-width="1" stroke-dasharray="2 3"/>')
    parts.append(f'<text x="{rx1-2:.1f}" y="{RY(inv_e)-4:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">1/e = {inv_e:.3f}</text>')
    frac_pts = " ".join(f"{RX(nn):.1f},{RY(optimal_fraction(nn)):.1f}" for nn in ns)
    prob_pts = " ".join(f"{RX(nn):.1f},{RY(optimal_cutoff(nn)[1]):.1f}" for nn in ns)
    parts.append(f'<polyline points="{frac_pts}" fill="none" stroke="#ff922b" stroke-width="2"/>')
    parts.append(f'<polyline points="{prob_pts}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    for nn in ns:
        parts.append(f'<circle cx="{RX(nn):.1f}" cy="{RY(optimal_fraction(nn)):.1f}" r="2" fill="#ff922b"/>')
        parts.append(f'<circle cx="{RX(nn):.1f}" cy="{RY(optimal_cutoff(nn)[1]):.1f}" r="2" fill="#06d6a0"/>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">n (log scale)</text>')
    # legend
    parts.append(f'<rect x="{rx0+8}" y="{ry1}" width="9" height="9" fill="#ff922b"/>'
                 f'<text x="{rx0+21}" y="{ry1+8}" fill="#e6edf3" font-size="9">look fraction</text>')
    parts.append(f'<rect x="{rx0+8}" y="{ry1+14}" width="9" height="9" fill="#06d6a0"/>'
                 f'<text x="{rx0+21}" y="{ry1+22}" fill="#e6edf3" font-size="9">win probability</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
