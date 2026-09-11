"""Demo: gambler's ruin -- the walk that ends at a wall.

Prints the ruin probability and expected duration for a fair game and a slightly unfair one
against a Monte-Carlo run, then draws the ruin-probability-versus-starting-stake curves for
several win probabilities and a few sample walk paths ending at 0 or N.

    python examples/gamblers_ruin_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gamblers_ruin import (ruin_probability, expected_duration,  # noqa: E402
                           _Rng, simulate)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    N = 100
    print(f"Gambler's ruin: bet $1 a round from i dollars, absorbed at 0 or N={N}\n")
    print(f"  {'p':>6}{'i':>6}{'ruin':>10}{'sim ruin':>10}{'duration':>12}{'sim dur':>10}")
    for p, i in ((0.50, 50), (0.50, 25), (0.49, 50), (0.60, 50)):
        rf, md = simulate(i, N, p, trials=6000, seed=3)
        print(f"  {p:>6.2f}{i:>6}{ruin_probability(i, N, p):>10.4f}{rf:>10.4f}"
              f"{expected_duration(i, N, p):>12.1f}{md:>10.1f}")

    print("\n  In a fair game the ruin chance is exactly 1 - i/N: your stake as a fraction of the")
    print("  table. But shift the odds a hair to p=0.49 and starting at the halfway mark the ruin")
    print("  chance leaps from 50% to 88% -- and against an infinitely rich house any p <= 1/2 is")
    print("  ruin with certainty. That asymmetry is why the house always wins.")

    _svg(os.path.join(outdir, "gamblers_ruin.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'gamblers_ruin.svg')}")


def _svg(path, w=760, h=380):
    N = 100
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Gambler\'s ruin: a tiny edge decides everything</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'ruin probability vs starting stake for N=100 (left); sample $1 walks to a wall (right)</text>',
    ]

    # left: ruin probability vs i for several p
    lx0, lx1 = 55, w // 2 - 20
    ly0, ly1 = h - 50, 60

    def LX(i):
        return lx0 + i / N * (lx1 - lx0)

    def LY(pr):
        return ly0 - pr * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    curves = [(0.60, "#06d6a0", "p=0.60 (favorable)"),
              (0.50, "#4dabf7", "p=0.50 (fair)"),
              (0.45, "#ff922b", "p=0.45"),
              (0.40, "#ff6b6b", "p=0.40 (unfavorable)")]
    for p, col, _ in curves:
        pts = " ".join(f"{LX(i):.1f},{LY(ruin_probability(i, N, p)):.1f}" for i in range(0, N + 1, 2))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
    for pr in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{lx0-6:.1f}" y="{LY(pr)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{pr:.1f}</text>')
    for i in (0, 50, 100):
        parts.append(f'<text x="{LX(i):.1f}" y="{ly0+16:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{i}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">starting stake i</text>')
    ly = ly1
    for p, col, lab in curves:
        parts.append(f'<rect x="{lx0+8}" y="{ly}" width="9" height="9" fill="{col}"/>'
                     f'<text x="{lx0+21}" y="{ly+8}" fill="#e6edf3" font-size="9">{lab}</text>')
        ly += 14

    # right: sample walk paths from i=50, p=0.49
    rx0, rx1 = w // 2 + 45, w - 20
    ry0, ry1 = h - 50, 60
    i0, p = 50, 0.49
    tmax = 2600

    def RX(t):
        return rx0 + t / tmax * (rx1 - rx0)

    def RY(pos):
        return ry0 - pos / N * (ry0 - ry1)

    # 0 and N walls
    parts.append(f'<line x1="{rx0}" y1="{RY(0):.1f}" x2="{rx1}" y2="{RY(0):.1f}" '
                 f'stroke="#ff6b6b" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<line x1="{rx0}" y1="{RY(N):.1f}" x2="{rx1}" y2="{RY(N):.1f}" '
                 f'stroke="#06d6a0" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{rx0+2:.1f}" y="{RY(0)-3:.1f}" fill="#ff6b6b" font-size="9">ruin (0)</text>')
    parts.append(f'<text x="{rx0+2:.1f}" y="{RY(N)+11:.1f}" fill="#06d6a0" font-size="9">target N</text>')
    colors = ["#b197fc", "#ffd43b", "#4dabf7", "#8338ec", "#ff922b"]
    for s in range(5):
        rng = _Rng(seed=10 + s * 7)
        pos = i0
        pts = [f"{RX(0):.1f},{RY(pos):.1f}"]
        t = 0
        while 0 < pos < N and t < tmax:
            pos += 1 if rng.random() < p else -1
            t += 1
            if t % 6 == 0 or pos in (0, N):
                pts.append(f"{RX(t):.1f},{RY(pos):.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[s]}" '
                     f'stroke-width="1.3" opacity="0.9"/>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">rounds -> (start i=50, p=0.49)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
