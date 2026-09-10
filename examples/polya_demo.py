"""Demo: Polya's random walk -- return probability across dimensions.

Prints the return and escape probabilities, expected origin visits, and simulated return
fractions by dimension, then draws the return probability dropping below 1 past two dimensions
-- the recurrent/transient boundary of Polya's theorem.

    python examples/polya_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from polya import (return_probability, is_recurrent, expected_visits,  # noqa: E402
                   escape_probability, simulate_return_fraction)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Polya's random walk: does an infinite lattice walk return to the origin?\n")
    print("  'A drunk man finds his way home; a drunk bird may get lost forever.'\n")
    print(f"  {'dim':>4}{'return prob':>13}{'escape prob':>13}{'exp. visits':>13}{'class':>12}")
    for d in (1, 2, 3, 4, 5):
        pv = expected_visits(d)
        vs = "inf" if math.isinf(pv) else f"{pv:.3f}"
        cls = "recurrent" if is_recurrent(d) else "transient"
        print(f"  {d:>4}{return_probability(d):>13.4f}{escape_probability(d):>13.4f}{vs:>13}{cls:>12}")

    print("\n  Simulated return fraction (finite walks, 3000 steps, 200 trials):")
    for d in (1, 2, 3):
        print(f"    {d}D  ->  {simulate_return_fraction(d, max_steps=3000, trials=200, seed=1)*100:.0f} % returned")

    print("\n  In 1D and 2D the walk is certain to return (and visits every site infinitely");
    print("  often); in 3D+ it can escape to infinity. The knife-edge is exactly two dimensions,")
    print("  because P(at origin) decays as n^(-d/2) -- summable (transient) only for d >= 3.")

    _svg(os.path.join(outdir, "polya.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'polya.svg')}")


def _svg(path, size=720, pad=80):
    dims = [1, 2, 3, 4, 5, 6]
    probs = [return_probability(d) for d in dims]

    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 44

    def X(d):
        return x0 + (d - 1) / (len(dims) - 1) * (x1 - x0)

    def Y(p):
        return y0 - p * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Polya&#39;s theorem: return probability vs dimension</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'recurrent (returns for sure) in 1D-2D, transient (can escape) in 3D and above</text>',
    ]

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.5"/>')
    for p in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{x0-6:.1f}" y="{Y(p)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{p:.1f}</text>')
        parts.append(f'<line x1="{x0}" y1="{Y(p):.1f}" x2="{x1}" y2="{Y(p):.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')

    # recurrent region shading (d<=2)
    parts.append(f'<rect x="{x0:.1f}" y="{y1:.1f}" width="{X(2)-x0:.1f}" height="{y0-y1:.1f}" '
                 f'fill="#06d6a0" opacity="0.06"/>')
    parts.append(f'<line x1="{X(2.5):.1f}" y1="{y0:.1f}" x2="{X(2.5):.1f}" y2="{y1:.1f}" '
                 f'stroke="#ff6b6b" stroke-width="1.2" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{X(1.5):.1f}" y="{y1+14:.1f}" fill="#06d6a0" font-size="11" '
                 f'text-anchor="middle">recurrent (p=1)</text>')
    parts.append(f'<text x="{X(4.5):.1f}" y="{y1+14:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">transient (p&lt;1)</text>')

    # bars
    bw = (x1 - x0) / len(dims) * 0.45
    for d, p in zip(dims, probs):
        col = "#06d6a0" if d <= 2 else "#4dabf7"
        parts.append(f'<rect x="{X(d)-bw/2:.1f}" y="{Y(p):.1f}" width="{bw:.1f}" '
                     f'height="{y0-Y(p):.1f}" fill="{col}" opacity="0.85"/>')
        parts.append(f'<text x="{X(d):.1f}" y="{Y(p)-6:.1f}" fill="{col}" font-size="10" '
                     f'text-anchor="middle">{p:.3f}</text>')
        parts.append(f'<text x="{X(d):.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{d}D</text>')

    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+32:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">lattice dimension</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
