"""Demo: Kitamasa -- the N-th term of a linear recurrence in O(k^2 log n).

Computes far-out terms of Fibonacci-like recurrences (including the 10^18-th, modulo a prime), shows
the characteristic-polynomial reduction, and contrasts the log-time cost with O(n) unrolling.

    python examples/kitamasa_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kitamasa import nth_term, nth_term_direct, characteristic_poly  # noqa: E402

MOD = 10 ** 9 + 7


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Kitamasa: the N-th linear-recurrence term without unrolling\n")

    recurrences = [
        ("Fibonacci", [1, 1], [0, 1]),
        ("Tribonacci", [1, 1, 1], [0, 0, 1]),
        ("Pell", [2, 1], [0, 1]),
        ("Perrin", [0, 1, 1], [3, 0, 2]),
    ]
    print(f"  {'sequence':12s} {'char poly':22s} {'term 30':>14}")
    for name, coeffs, init in recurrences:
        char = characteristic_poly(coeffs)
        poly_str = _poly_str(char)
        t30 = nth_term(coeffs, init, 30)
        print(f"  {name:12s} {poly_str:22s} {t30:>14}")
        assert t30 == nth_term_direct(coeffs, init, 30)

    print("\n  exact small Fibonacci (verified against O(n) unrolling):")
    for n in (10, 50, 100):
        print(f"    F({n}) = {nth_term([1, 1], [0, 1], n)}")

    print(f"\n  enormous terms modulo p = {MOD} (impossible by unrolling):")
    for n in (10 ** 6, 10 ** 12, 10 ** 18):
        print(f"    F({n}) mod p = {nth_term([1, 1], [0, 1], n, mod=MOD)}")

    print("\n  Kitamasa computes x^n mod c(x) by square-and-multiply (each reduction O(k^2)), then dots")
    print("  the resulting coefficient vector with the initial terms. So the N-th term costs only")
    print("  O(k^2 log n) -- for n = 10^18 that is a few dozen polynomial multiplies, versus a")
    print("  quintillion additions to unroll. It beats even the O(k^3 log n) companion-matrix power.")

    _svg(os.path.join(outdir, "kitamasa.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'kitamasa.svg')}")


def _poly_str(char):
    # char is [1, -c1, -c2, ...] high-degree-first
    k = len(char) - 1
    terms = []
    for i, c in enumerate(char):
        p = k - i
        if c == 0:
            continue
        cs = f"{c:+d}" if p == 0 else (f"{c:+d}x^{p}" if p > 1 else f"{c:+d}x")
        if i == 0:
            cs = "x^%d" % p if p > 1 else ("x" if p == 1 else "1")
        terms.append(cs)
    return " ".join(terms).replace("+", "+ ").replace("-", "- ").lstrip()


def _svg(path, width=740, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Cost of the N-th recurrence term: unrolling O(n) vs Kitamasa O(log n)</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'operations vs n (log-log): unrolling is linear, Kitamasa is logarithmic -- a chasm at large n'
        f'</text>',
    ]
    ox, oy = 70, 320
    pw, ph = width - 130, 240
    ns = [10 ** (i / 3) for i in range(3, 55)]      # n from ~10 to ~10^18

    def px(n):
        return ox + (math.log10(n) - math.log10(ns[0])) / (math.log10(ns[-1]) - math.log10(ns[0])) * pw

    # costs: unroll ~ n ; kitamasa ~ log2(n) * k^2 (k=2)
    unroll = [math.log10(n) for n in ns]
    kita = [math.log10(max(1, math.log2(n) * 4)) for n in ns]
    allc = unroll + kita
    cmin, cmax = min(allc), max(allc)

    def py(c):
        return oy - (c - cmin) / (cmax - cmin) * ph

    for data, col, label in [(unroll, "#ff6b6b", "unroll O(n)"),
                            (kita, "#06d6a0", "Kitamasa O(log n)")]:
        pts = " ".join(f"{px(n):.1f},{py(c):.1f}" for n, c in zip(ns, data))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        parts.append(f'<text x="{px(ns[-1])-2:.0f}" y="{py(data[-1])+ (14 if label.startswith("Kita") else -6):.0f}" '
                     f'fill="{col}" font-size="11" text-anchor="end">{label}</text>')

    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d"/>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+30}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">term index n (log scale, up to 10^18) -></text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
