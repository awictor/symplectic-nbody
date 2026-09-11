"""Demo: the Monty Hall problem -- why switching doors wins.

Prints the stay/switch win probabilities for the classic game and the knowledge-free variant
against a Monte-Carlo play, then draws the classic stay-vs-switch bars and how the switch
advantage grows as the game scales to N doors with all-but-one goat door revealed.

    python examples/monty_hall_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monty_hall import (stay_probability, switch_probability,  # noqa: E402
                        random_host_switch_probability, simulate)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Monty Hall: the knowing host's reveal shifts the odds -- switch and win 2/3\n")
    print(f"  {'strategy':>22}{'theory':>9}{'sim':>9}")
    print(f"  {'stay (classic)':>22}{stay_probability(3):>9.3f}"
          f"{simulate('stay', trials=40000, seed=5):>9.3f}")
    print(f"  {'switch (classic)':>22}{switch_probability(3, 1):>9.3f}"
          f"{simulate('switch', trials=40000, seed=5):>9.3f}")
    print(f"  {'switch (random host)':>22}{random_host_switch_probability(3):>9.3f}"
          f"{simulate('switch', trials=60000, seed=5, informed_host=False):>9.3f}")

    print("\n  Scaling up: N doors, the host opens all but one other goat door, you switch:")
    print(f"  {'doors N':>10}{'stay 1/N':>11}{'switch':>10}")
    for N in (3, 10, 50, 100):
        print(f"  {N:>10}{stay_probability(N):>11.4f}{switch_probability(N, N - 2):>10.4f}")
    print("\n  Your first pick is right only 1/N; the host's knowing reveal piles the whole")
    print("  remaining (N-1)/N onto the last closed door. With 100 doors, switching wins 99%.")
    print("  (If the host opened doors blindly, the effect vanishes -- it is his knowledge.)")

    _svg(os.path.join(outdir, "monty_hall.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'monty_hall.svg')}")


def _svg(path, w=760, h=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'The Monty Hall problem: always switch</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'classic 3-door win rates (left); switching wins (N-1)/N as the game scales (right)</text>',
    ]

    # left: three bars -- stay, switch, random-host switch
    lx0, lx1 = 55, w // 2 - 20
    ly0, ly1 = h - 55, 65
    bars = [("stay", stay_probability(3), "#ff6b6b"),
            ("switch", switch_probability(3, 1), "#06d6a0"),
            ("switch\n(blind host)", random_host_switch_probability(3), "#ff922b")]

    def LY(p):
        return ly0 - p * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    for p in (0.0, 1 / 3, 2 / 3, 1.0):
        parts.append(f'<line x1="{lx0}" y1="{LY(p):.1f}" x2="{lx1}" y2="{LY(p):.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{lx0-6:.1f}" y="{LY(p)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{p:.2f}</text>')
    n = len(bars)
    slot = (lx1 - lx0) / n
    for i, (lab, p, col) in enumerate(bars):
        cx = lx0 + (i + 0.5) * slot
        bw = slot * 0.5
        parts.append(f'<rect x="{cx-bw/2:.1f}" y="{LY(p):.1f}" width="{bw:.1f}" '
                     f'height="{ly0-LY(p):.1f}" fill="{col}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{LY(p)-6:.1f}" fill="{col}" font-size="11" '
                     f'text-anchor="middle">{p:.3f}</text>')
        for j, line in enumerate(lab.split("\n")):
            parts.append(f'<text x="{cx:.1f}" y="{ly0+15+j*11:.1f}" fill="#8b949e" font-size="9" '
                         f'text-anchor="middle">{line}</text>')

    # right: switch probability vs N (host opens all but one other)
    rx0, rx1 = w // 2 + 45, w - 30
    ry0, ry1 = h - 55, 65
    ns = [3, 4, 5, 7, 10, 15, 25, 40, 70, 100]
    lo, hi = math.log(ns[0]), math.log(ns[-1])

    def RX(N):
        return rx0 + (math.log(N) - lo) / (hi - lo) * (rx1 - rx0)

    def RY(p):
        return ry0 - p * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{RY(1.0):.1f}" x2="{rx1}" y2="{RY(1.0):.1f}" '
                 f'stroke="#21262d" stroke-width="1"/>')
    sw = " ".join(f"{RX(N):.1f},{RY(switch_probability(N, N - 2)):.1f}" for N in ns)
    st = " ".join(f"{RX(N):.1f},{RY(stay_probability(N)):.1f}" for N in ns)
    parts.append(f'<polyline points="{sw}" fill="none" stroke="#06d6a0" stroke-width="2.5"/>')
    parts.append(f'<polyline points="{st}" fill="none" stroke="#ff6b6b" stroke-width="2"/>')
    for N in ns:
        parts.append(f'<circle cx="{RX(N):.1f}" cy="{RY(switch_probability(N, N-2)):.1f}" r="2.5" fill="#06d6a0"/>')
    for p in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{rx0-6:.1f}" y="{RY(p)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{p:.1f}</text>')
    for N in (3, 10, 100):
        parts.append(f'<text x="{RX(N):.1f}" y="{ry0+15:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{N}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">doors N (log)</text>')
    parts.append(f'<text x="{RX(ns[-1]):.1f}" y="{RY(switch_probability(100,98))-6:.1f}" fill="#06d6a0" '
                 f'font-size="9" text-anchor="end">switch (N-1)/N</text>')
    parts.append(f'<text x="{RX(ns[-1]):.1f}" y="{RY(stay_probability(100))-6:.1f}" fill="#ff6b6b" '
                 f'font-size="9" text-anchor="end">stay 1/N</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
