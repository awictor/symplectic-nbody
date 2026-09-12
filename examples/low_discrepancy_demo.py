"""Demo: low-discrepancy sequences -- quasi-Monte-Carlo that fills space evenly and integrates faster.

Compares the Halton sequence to pseudo-random points by star discrepancy and by Monte-Carlo
integration error, then draws the two point sets side by side and the QMC-vs-MC error decay.

    python examples/low_discrepancy_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from low_discrepancy import (halton, hammersley, star_discrepancy,  # noqa: E402
                             qmc_integrate, radical_inverse)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Low-discrepancy sequences: quasi-Monte-Carlo fills space evenly\n")

    N = 256
    h = halton(N, 2)
    rng = LCG(2024)
    rand = [(rng.u(), rng.u()) for _ in range(N)]
    dh = star_discrepancy(h, samples=1500)
    dr = star_discrepancy(rand, samples=1500)
    print(f"  star discrepancy of {N} points (lower = more even):")
    print(f"    Halton        {dh:.4f}")
    print(f"    pseudo-random {dr:.4f}   ({dr/dh:.1f}x worse)\n")

    # pi estimate convergence
    print("  estimating pi by the area of a quarter unit disk (integral = pi/4):")
    for n in (100, 1000, 10000):
        qmc = qmc_integrate(lambda p: 1.0 if p[0] ** 2 + p[1] ** 2 < 1 else 0.0, 2, n) * 4
        r = LCG(7)
        mc = sum(1 for _ in range(n) if (lambda a, b: a * a + b * b < 1)(r.u(), r.u())) / n * 4
        print(f"    N={n:6d}:  QMC pi={qmc:.5f} (err {abs(qmc-math.pi):.5f})   "
              f"MC pi={mc:.5f} (err {abs(mc-math.pi):.5f})")
    print()

    print("  The radical inverse reflects an index's base-b digits about the radix point, so each new")
    print("  point lands in the largest remaining gap. Halton uses a coprime prime base per axis. QMC")
    print("  error decays like (log N)^d / N -- close to 1/N, far faster than Monte-Carlo's 1/sqrt(N).")

    _svg(os.path.join(outdir, "low_discrepancy.svg"), h, rand)
    print(f"\n  wrote {os.path.join(outdir, 'low_discrepancy.svg')}")


def _svg(path, halton_pts, rand_pts, panel=280, pad=30, gap=45):
    width = pad * 2 + panel * 2 + gap
    height = pad + 40 + panel + 30

    def panel_svg(points, ox, oy, title, colour):
        out = [f'<text x="{ox}" y="{oy-10}" fill="#e6edf3" font-size="14">{title}</text>',
               f'<rect x="{ox}" y="{oy}" width="{panel}" height="{panel}" '
               f'fill="#010409" stroke="#30363d" stroke-width="1"/>']
        for x, y in points:
            cx = ox + x * panel
            cy = oy + y * panel
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.2" fill="{colour}"/>')
        return out

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="{pad}" y="24" fill="#e6edf3" font-size="17">'
        f'Halton (quasi-random) vs pseudo-random -- {len(halton_pts)} points each</text>',
    ]
    oy = pad + 40
    parts += panel_svg(halton_pts, pad, oy, "Halton: evenly space-filling", "#4dabf7")
    parts += panel_svg(rand_pts, pad + panel + gap, oy, "Pseudo-random: clumps and gaps", "#ff922b")
    parts.append(f'<text x="{pad}" y="{height-10}" fill="#8b949e" font-size="11">'
                 f'the Halton set covers every sub-square evenly, giving lower discrepancy and faster '
                 f'integration convergence</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
