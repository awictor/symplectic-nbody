"""Demo: Egyptian fractions -- the greedy Fibonacci-Sylvester algorithm and Sylvester's sequence.

Decomposes several fractions into distinct unit fractions by the greedy algorithm, shows the Engel
expansion, and displays Sylvester's sequence whose reciprocals race toward 1. Draws the greedy
decomposition of a fraction as a stack of shrinking unit-fraction bars.

    python examples/egyptian_fraction_demo.py [output_dir]
"""

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from egyptian_fraction import (  # noqa: E402
    greedy_egyptian, egyptian_sum, engel_expansion, sylvester_sequence, sylvester_reciprocal_sum,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Egyptian fractions: every rational as a sum of distinct unit fractions\n")

    print(f"  greedy Fibonacci-Sylvester decompositions:")
    for num, den in [(2, 3), (3, 7), (5, 6), (4, 13), (5, 121), (2, 2011)]:
        denoms = greedy_egyptian(num, den)
        terms = " + ".join(f"1/{d}" for d in denoms)
        print(f"    {num}/{den} = {terms}")
        assert egyptian_sum(denoms) == Fraction(num, den)

    print(f"\n  Engel expansions (x = 1/a1 + 1/(a1 a2) + ..., a non-decreasing):")
    for num, den in [(3, 7), (5, 12), (7, 15)]:
        print(f"    {num}/{den}: a = {engel_expansion(num, den)}")

    print(f"\n  Sylvester's sequence (each term = 1 + product of all previous):")
    seq = sylvester_sequence(6)
    print(f"    {seq}")
    print(f"    sum of reciprocals races toward 1:")
    for n in range(1, 6):
        s = sylvester_reciprocal_sum(n)
        print(f"      {n} terms: {s}  (1 - {1 - s})")

    print(f"\n  The greedy numerator strictly shrinks each step (Fibonacci's 1202 termination proof),")
    print(f"  so every fraction in (0,1) has a finite Egyptian expansion with distinct denominators.")

    _svg(os.path.join(outdir, "egyptian_fraction.svg"), 4, 13)
    print(f"\n  wrote {os.path.join(outdir, 'egyptian_fraction.svg')}")


def _svg(path, num, den, width=760, height=360):
    denoms = greedy_egyptian(num, den)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'{num}/{den} = {" + ".join(f"1/{d}" for d in denoms)} (unit fractions, distinct)</text>',
    ]
    ox, oy, ow = 50, 70, width - 100
    # total bar represents num/den; each unit fraction is a segment of width proportional to 1/d
    total = float(num) / den
    x = ox
    colors = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#b197fc", "#ff6b6b"]
    barh = 60
    for i, d in enumerate(denoms):
        frac = (1.0 / d) / total
        w = ow * frac
        col = colors[i % len(colors)]
        parts.append(f'<rect x="{x:.1f}" y="{oy}" width="{w:.1f}" height="{barh}" fill="{col}" '
                     f'stroke="#0d1117"/>')
        if w > 30:
            parts.append(f'<text x="{x + w/2:.1f}" y="{oy + barh/2 + 4:.1f}" fill="#0d1117" '
                         f'font-size="12" text-anchor="middle">1/{d}</text>')
        else:
            parts.append(f'<text x="{x + w/2:.1f}" y="{oy - 6:.1f}" fill="{col}" '
                         f'font-size="9" text-anchor="middle">1/{d}</text>')
        x += w
    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{barh}" fill="none" stroke="#30363d"/>')
    parts.append(f'<text x="{ox}" y="{oy + barh + 24:.0f}" fill="#8b949e" font-size="10">'
                 f'the full bar is {num}/{den}; each coloured segment is one distinct unit fraction, '
                 f'shrinking left to right</text>')
    # Sylvester reciprocal-sum convergence below
    sy = oy + barh + 60
    parts.append(f'<text x="{ox}" y="{sy - 8:.0f}" fill="#8b949e" font-size="11">'
                 f'Sylvester reciprocal sums approaching 1:</text>')
    seq = sylvester_sequence(6)
    prev = 0.0
    for i in range(1, 7):
        s = float(sylvester_reciprocal_sum(i))
        bx = ox + ow * s
        parts.append(f'<line x1="{ox}" y1="{sy + i*18:.0f}" x2="{bx:.1f}" y2="{sy + i*18:.0f}" '
                     f'stroke="#06d6a0" stroke-width="6"/>')
        parts.append(f'<text x="{bx + 6:.1f}" y="{sy + i*18 + 4:.0f}" fill="#8b949e" font-size="9">'
                     f'{i} terms</text>')
    parts.append(f'<line x1="{ox+ow:.0f}" y1="{sy:.0f}" x2="{ox+ow:.0f}" y2="{sy + 7*18:.0f}" '
                 f'stroke="#ff6b6b" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{ox+ow-4:.0f}" y="{sy - 8:.0f}" fill="#ff6b6b" font-size="9" '
                 f'text-anchor="end">1</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
