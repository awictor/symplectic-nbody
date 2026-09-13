"""Demo: fast-doubling Fibonacci -- the millionth term, Pisano periods, and Zeckendorf sums.

Computes enormous Fibonacci numbers in O(log n), verifies Cassini's identity and the GCD property,
shows how the Pisano period lets F_n mod m be found for astronomical n, and displays Zeckendorf
representations. Draws the Pisano periods pi(m).

    python examples/fibonacci_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fibonacci import (  # noqa: E402
    fibonacci, lucas, fib_mod, pisano_period, zeckendorf, cassini, fib_index,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Fibonacci by fast doubling: O(log n), so million-digit terms are instant\n")

    for n in [10, 100, 1000]:
        f = fibonacci(n)
        print(f"  F_{n} = {f if f < 1e15 else str(f)[:20] + '... (' + str(len(str(f))) + ' digits)'}")
    big = fibonacci(1_000_000)
    # digit count without str() (Python caps int->str conversion): floor(log10) + 1 via bit_length
    digits = int(big.bit_length() * math.log10(2)) + 1
    print(f"  F_1000000 has ~{digits} digits (computed via ~20 big-integer multiplies)\n")

    # identities
    print(f"  Cassini's identity F_(n-1)F_(n+1) - F_n^2 = (-1)^n:")
    for n in [5, 6, 7, 8]:
        print(f"    n={n}: {cassini(n):+d}")
    print(f"\n  GCD property gcd(F_m, F_n) = F_gcd(m,n):")
    for m, n in [(12, 18), (15, 25), (14, 21)]:
        g = math.gcd(fibonacci(m), fibonacci(n))
        print(f"    gcd(F_{m}, F_{n}) = {g} = F_{math.gcd(m, n)} = {fibonacci(math.gcd(m, n))}")

    # Pisano period
    print(f"\n  Pisano period pi(m) (period of F_n mod m) lets F_n mod m work for huge n:")
    for m in [2, 3, 10, 100, 1000]:
        p = pisano_period(m)
        print(f"    pi({m}) = {p}")
    n_huge = 10 ** 100
    print(f"  F_(10^100) mod 1000 = {fib_mod(n_huge, 1000)}  (via the Pisano cycle)")

    # Zeckendorf
    print(f"\n  Zeckendorf representations (unique non-consecutive Fibonacci sums):")
    for n in [17, 100, 1000]:
        rep = zeckendorf(n)
        print(f"    {n} = {' + '.join(map(str, rep))}")

    _svg(os.path.join(outdir, "fibonacci.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'fibonacci.svg')}")


def _svg(path, width=760, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Pisano period pi(m): the length of the cycle of F_n mod m</text>',
    ]
    ox, oy, ow, oh = 55, 55, width - 100, height - 100
    ms = list(range(2, 101))
    periods = [pisano_period(m) for m in ms]
    pmax = max(periods)
    mmax = max(ms)

    def px(m):
        return ox + ow * (m - 2) / (mmax - 2)

    def py(p):
        return oy + oh * (1 - p / pmax)

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    for m, p in zip(ms, periods):
        x = px(m)
        h = oy + oh - py(p)
        # highlight m dividing into the 6m bound cases
        color = "#ffd43b" if p == 6 * m else "#4dabf7"
        parts.append(f'<rect x="{x-2:.1f}" y="{py(p):.1f}" width="3.5" height="{h:.1f}" fill="{color}"/>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+22:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">m (yellow = the maximal pi(m) = 6m cases)</text>')
    parts.append(f'<text x="{ox-6}" y="{oy+6}" fill="#8b949e" font-size="9" text-anchor="end">{pmax}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
