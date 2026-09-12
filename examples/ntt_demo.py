"""Demo: the number-theoretic transform -- exact integer convolution, no rounding.

Multiplies polynomials and big integers exactly via the NTT (the FFT in modular arithmetic), contrasts
its exactness with the rounding of a complex FFT, and shows the O(n log n) win over schoolbook. Draws
a polynomial-product coefficient bar chart.

    python examples/ntt_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ntt import convolve, poly_multiply, multiply_big_integers, schoolbook_convolve, MOD  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Number-theoretic transform: FFT in modular arithmetic -> EXACT integer convolution\n")
    print(f"  working modulo the NTT-friendly prime p = {MOD} = 119 * 2^23 + 1\n")

    a = [1, 2, 3, 4]
    b = [5, 6, 7]
    prod = poly_multiply(a, b)
    print("  polynomial multiply (1 + 2x + 3x^2 + 4x^3)(5 + 6x + 7x^2):")
    print(f"    NTT      : {prod}")
    print(f"    schoolbook: {schoolbook_convolve(a, b)}")
    print(f"    exact match: {prod == schoolbook_convolve(a, b)}\n")

    print("  big-integer multiplication by digit convolution + carry:")
    for x, y in [(12345, 6789), (2 ** 64, 3 ** 40)]:
        got = multiply_big_integers(x, y)
        print(f"    {x} * {y} = {got}")
        print(f"      matches Python's exact bignum: {got == x * y}")

    huge = multiply_big_integers(10 ** 100 - 1, 10 ** 100 - 1)
    print(f"\n  (10^100 - 1)^2 has {len(str(huge))} digits, computed exactly: {huge == (10**100-1)**2}")

    print("\n  Why exact? The FFT uses complex roots e^(2 pi i / n) and carries floating-point error;")
    print("  the NTT replaces them with a primitive n-th root of unity modulo p, so every butterfly")
    print("  is an exact integer operation. Transform, multiply pointwise, inverse-transform -- the")
    print("  convolution theorem holds identically, but the answer is exact mod p (and the true")
    print("  integer whenever the values stay below p). This powers arbitrary-precision multiplication.")

    _svg(os.path.join(outdir, "ntt.svg"), a, b, prod)
    print(f"\n  wrote {os.path.join(outdir, 'ntt.svg')}")


def _svg(path, a, b, prod, width=760, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Exact polynomial product via the NTT</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'coefficients of (blue) x (green) = (yellow), each an exact integer -- no rounding</text>',
    ]

    def bars(coeffs, y0, col, label):
        mx = max(coeffs) if coeffs else 1
        cell = 46
        ox = 130
        parts.append(f'<text x="{ox-12}" y="{y0+18}" fill="{col}" font-size="12" '
                     f'text-anchor="end">{label}</text>')
        for i, c in enumerate(coeffs):
            x = ox + i * cell
            h = 40 * c / mx
            parts.append(f'<rect x="{x}" y="{y0+40-h:.1f}" width="{cell-6}" height="{h:.1f}" '
                         f'fill="{col}"/>')
            parts.append(f'<text x="{x+(cell-6)/2:.0f}" y="{y0+54:.0f}" fill="#8b949e" '
                         f'font-size="10" text-anchor="middle">{c}</text>')

    bars(a, 90, "#4dabf7", "A")
    bars(b, 170, "#06d6a0", "B")
    bars(prod, 260, "#ffd43b", "A*B")

    parts.append(f'<text x="130" y="{330}" fill="#8b949e" font-size="11">'
                 f'each product coefficient is sum of a[i]*b[j] over i+j=k -- convolution, done exactly'
                 f'</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
