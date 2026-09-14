"""Demo: Fermat's two-square theorem, Cornacchia's construction, and Lagrange's four squares.

Shows which numbers are sums of two squares and why (the 3-mod-4 prime rule), constructs
representations with Cornacchia + the Brahmagupta-Fibonacci identity, and demonstrates Lagrange's
theorem that every integer is a sum of four squares. Draws which n <= 100 are sums of 1, 2, 3, or 4
squares.

    python examples/sum_of_squares_demo.py [output_dir]
"""

import os
import sys
from math import isqrt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sum_of_squares import (  # noqa: E402
    is_sum_of_two_squares, two_squares, cornacchia, four_squares,
)


def _min_squares(n):
    """Minimum number of squares summing to n (1, 2, 3, or 4 by Lagrange)."""
    if n == 0:
        return 0
    r = isqrt(n)
    if r * r == n:
        return 1
    if is_sum_of_two_squares(n):
        return 2
    # Legendre three-square theorem: n needs 4 iff n = 4^a (8b+7)
    m = n
    while m % 4 == 0:
        m //= 4
    if m % 8 == 7:
        return 4
    return 3


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Sums of squares: Fermat's two-square theorem and Lagrange's four-square theorem\n")

    print(f"  primes as sums of two squares (Fermat: p = 2 or p = 1 mod 4):")
    print(f"    {'prime':>6}{'mod 4':>7}{'a^2 + b^2':>16}")
    for p in [2, 5, 13, 17, 29, 37, 41, 3, 7, 11, 19]:
        if is_sum_of_two_squares(p):
            a, b = cornacchia(p) if p % 4 == 1 or p == 2 else (0, 0)
            print(f"    {p:>6}{p % 4:>7}   {a}^2 + {b}^2 = {a*a+b*b}")
        else:
            print(f"    {p:>6}{p % 4:>7}   (not a sum of two squares)")

    print(f"\n  Cornacchia + Brahmagupta-Fibonacci build composite representations:")
    for n in [325, 1105, 5525]:
        r = two_squares(n)
        print(f"    {n} = {r[0]}^2 + {r[1]}^2 = {r[0]**2 + r[1]**2}")

    print(f"\n  Lagrange: EVERY integer is a sum of four squares (the 4^a(8b+7) cases need all four):")
    for n in [7, 15, 23, 31, 127, 4 * 7]:
        rep = four_squares(n)
        nonzero = sum(1 for x in rep if x != 0)
        print(f"    {n} = {' + '.join(f'{x}^2' for x in rep if x != 0)}  ({nonzero} nonzero squares)")

    _svg(os.path.join(outdir, "sum_of_squares.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'sum_of_squares.svg')}")


def _svg(path, width=760, height=420):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Minimum number of squares summing to n (1..100): Lagrange caps it at 4</text>',
    ]
    colors = {1: "#06d6a0", 2: "#4dabf7", 3: "#ffd43b", 4: "#ff6b6b"}
    cols = 10
    cell = 60
    ox, oy = 60, 60
    for n in range(1, 101):
        k = _min_squares(n)
        i = (n - 1) // cols
        j = (n - 1) % cols
        x = ox + j * cell
        y = oy + i * cell
        parts.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{cell-4}" height="{cell-4}" '
                     f'fill="{colors[k]}" rx="4"/>')
        parts.append(f'<text x="{x+cell/2-2:.0f}" y="{y+cell/2-2:.0f}" fill="#0d1117" font-size="12" '
                     f'text-anchor="middle">{n}</text>')
        parts.append(f'<text x="{x+cell/2-2:.0f}" y="{y+cell/2+12:.0f}" fill="#0d1117" font-size="9" '
                     f'text-anchor="middle">{k}sq</text>')
    # legend
    lx = ox
    for k in (1, 2, 3, 4):
        parts.append(f'<rect x="{lx}" y="{oy+6*cell+6}" width="14" height="14" fill="{colors[k]}"/>')
        label = {1: "perfect square", 2: "two squares", 3: "three squares", 4: "four (=4^a(8b+7))"}[k]
        parts.append(f'<text x="{lx+18}" y="{oy+6*cell+18}" fill="#8b949e" font-size="9">{label}</text>')
        lx += 175
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
