"""Demo: Karatsuba and Toom-Cook -- multiplying big numbers with fewer sub-products.

Multiplies big integers via Karatsuba and Toom-3, verifies them against Python's exact bignum, and
shows how the number of recursive sub-multiplications (and hence the complexity exponent) drops from
schoolbook's 4-per-split to Karatsuba's 3 to Toom-3's 5-of-9.

    python examples/karatsuba_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from karatsuba import karatsuba, toom3, poly_multiply_karatsuba, _poly_schoolbook  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Karatsuba & Toom-Cook: fast multiplication by fewer sub-products\n")

    for x, y in [(12345678, 87654321), (2 ** 128, 3 ** 80)]:
        k = karatsuba(x, y)
        t = toom3(x, y)
        print(f"  {x} * {y}")
        print(f"    Karatsuba = {k}")
        print(f"    Toom-3    = {t}")
        print(f"    both match Python: {k == t == x * y}\n")

    big = karatsuba(10 ** 300 - 1, 10 ** 300 - 1)
    print(f"  (10^300 - 1)^2 has {len(str(big))} digits, computed exactly: {big == (10**300-1)**2}")

    a = [1, 2, 3, 4, 5]
    b = [6, 7, 8]
    print(f"\n  polynomial multiply {a} x {b}:")
    print(f"    Karatsuba  = {poly_multiply_karatsuba(a, b)}")
    print(f"    schoolbook = {_poly_schoolbook(a, b)}")

    print("\n  complexity comparison (sub-multiplications per level, and the resulting exponent):")
    methods = [("schoolbook", 4, 2, math.log(4, 2)),
               ("Karatsuba", 3, 2, math.log(3, 2)),
               ("Toom-3", 5, 3, math.log(5, 3))]
    print(f"    {'method':12s} {'mults':>6} {'splits':>7} {'exponent O(n^e)':>16}")
    for name, mults, splits, exp in methods:
        print(f"    {name:12s} {mults:>6} {splits:>7} {exp:>16.4f}")
    print("\n  Karatsuba: split each number in 2, and the middle cross-term x1*y0 + x0*y1 comes from")
    print("  ONE product (x1+x0)(y1+y0) minus the two you already have -- 3 mults, not 4. Toom-3 splits")
    print("  in 3 and interpolates a degree-4 product from 5 evaluations, not 9. Each cut of the")
    print("  exponent matters enormously for the thousand-digit numbers inside bignum libraries.")

    _svg(os.path.join(outdir, "karatsuba.svg"), methods)
    print(f"\n  wrote {os.path.join(outdir, 'karatsuba.svg')}")


def _svg(path, methods, width=720, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Cost of multiplying two n-digit numbers, by method</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'runtime O(n^e); lower exponent wins big for large n (curves are n^e, log scale in n)</text>',
    ]

    # plot n^e for each method over a range of n, log-log
    ox, oy = 70, 330
    pw, ph = width - 120, 250
    import math as m
    ns = [10 ** (i / 8) for i in range(8, 33)]     # n from ~3 to ~10^4
    colors = ["#ff6b6b", "#4dabf7", "#06d6a0"]

    def px(n):
        return ox + (m.log10(n) - m.log10(ns[0])) / (m.log10(ns[-1]) - m.log10(ns[0])) * pw

    # normalise costs to a common log range
    all_costs = []
    for (_, _, _, e) in methods:
        for n in ns:
            all_costs.append(e * m.log10(n))
    cmin, cmax = min(all_costs), max(all_costs)

    def py(cost_log):
        return oy - (cost_log - cmin) / (cmax - cmin) * ph

    for (name, _, _, e), col in zip(methods, colors):
        pts = " ".join(f"{px(n):.1f},{py(e * m.log10(n)):.1f}" for n in ns)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        parts.append(f'<text x="{px(ns[-1])-2:.0f}" y="{py(e * m.log10(ns[-1]))-6:.0f}" '
                     f'fill="{col}" font-size="11" text-anchor="end">{name} (e={e:.3f})</text>')

    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d"/>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+30}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">number size n (log scale) -></text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
