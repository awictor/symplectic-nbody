"""Demo: Dixon's factorization -- manufacturing a congruence of squares to crack a semiprime.

Factors a semiprime by Dixon's method, showing the factor base, a few smooth relations x^2 mod N and
their exponent vectors, the GF(2) dependency that combines them into a perfect square, and the gcd
that extracts the factor. Draws the exponent-vector matrix over the factor base.

    python examples/dixon_demo.py [output_dir]
"""

import os
import sys
from math import isqrt, gcd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dixon import factor_base, _smooth_vector, dixon_factor, factorize  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Dixon's factorization: a congruence of squares x^2 = y^2 (mod N) gives a factor\n")

    N = 8051   # = 83 * 97
    base = factor_base(N)
    print(f"  N = {N} = 83 x 97, factor base = {base}\n")

    # collect a handful of smooth relations for display
    r = isqrt(N)
    print(f"  smooth relations x^2 mod N that factor over the base:")
    print(f"    {'x':>5}{'x^2 mod N':>12}   exponent vector over base")
    rels = []
    x = r + 1
    while len(rels) < 8 and x < N:
        val = (x * x) % N
        vec = _smooth_vector(val, base)
        if vec is not None and val != 0:
            rels.append((x, val, vec))
            print(f"    {x:>5}{val:>12}   {vec}")
        x += 1

    factor = dixon_factor(N, seed=1)
    print(f"\n  a GF(2) combination of these relations gives X^2 = Y^2 (mod N),")
    print(f"  and gcd(X - Y, N) = {factor}  ->  N = {factor} x {N // factor}")
    print(f"\n  full factorization: {factorize(N)}")
    print(f"\n  This is the quadratic sieve's engine: find smooth squares, combine their exponent")
    print(f"  parities to zero via linear algebra over GF(2), read off a factor by gcd.")

    _svg(os.path.join(outdir, "dixon.svg"), base, rels)
    print(f"\n  wrote {os.path.join(outdir, 'dixon.svg')}")


def _svg(path, base, rels, width=760, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Smooth-relation exponent vectors mod 2 over the factor base (Dixon)</text>',
    ]
    m = len(base)
    n = len(rels)
    if n == 0:
        parts.append("</svg>")
        open(path, "w").write("\n".join(parts))
        return
    ox, oy = 120, 60
    cw = min(50, (width - ox - 40) / m)
    ch = min(30, (height - oy - 60) / n)

    # column headers = base primes
    for j, p in enumerate(base):
        parts.append(f'<text x="{ox + j*cw + cw/2:.0f}" y="{oy-8:.0f}" fill="#4dabf7" font-size="11" '
                     f'text-anchor="middle">{p}</text>')
    for i, (x, val, vec) in enumerate(rels):
        y = oy + i * ch
        parts.append(f'<text x="{ox-10:.0f}" y="{y+ch/2+4:.0f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="end">{x}^2={val}</text>')
        for j in range(m):
            parity = vec[j] % 2
            col = "#ffd43b" if parity else "#161b22"
            xx = ox + j * cw
            parts.append(f'<rect x="{xx:.1f}" y="{y:.1f}" width="{cw-2:.1f}" height="{ch-2:.1f}" '
                         f'fill="{col}" stroke="#30363d"/>')
            parts.append(f'<text x="{xx+cw/2-1:.0f}" y="{y+ch/2+4:.0f}" '
                         f'fill="{"#0d1117" if parity else "#484f58"}" font-size="10" '
                         f'text-anchor="middle">{parity}</text>')
    parts.append(f'<text x="{width/2:.0f}" y="{height-16}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">yellow = odd exponent; a subset XORing to all-zero is a perfect square</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
