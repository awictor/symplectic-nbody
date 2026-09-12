"""Demo: the Fast Walsh-Hadamard transform -- convolution over XOR instead of addition.

Shows XOR convolution as the distribution of the XOR of two random 3-bit values, compares the three
Boolean convolutions (XOR / OR / AND) on the same inputs, and draws the resulting distributions.

    python examples/walsh_hadamard_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from walsh_hadamard import (xor_convolution, or_convolution, and_convolution,  # noqa: E402
                            brute_convolution)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Fast Walsh-Hadamard transform: the FFT for XOR\n")

    # two probability distributions over 3-bit values 0..7
    a = [4, 3, 2, 1, 1, 1, 0, 0]     # process A favours small values
    b = [0, 0, 1, 2, 3, 3, 2, 1]     # process B favours mid/large values
    sa, sb = sum(a), sum(b)

    print("  Two independent processes each emit a 3-bit value with these weights:")
    print(f"    A (sum {sa}): {a}")
    print(f"    B (sum {sb}): {b}\n")

    xor = xor_convolution(a, b)
    orc = or_convolution(a, b)
    andc = and_convolution(a, b)

    print("  Distribution of the COMBINED value under each bitwise operation:")
    print(f"    value:      {list(range(8))}")
    print(f"    A XOR B:    {xor}")
    print(f"    A OR  B:    {orc}")
    print(f"    A AND B:    {andc}")
    print(f"\n  Each total mass = {sa}*{sb} = {sa*sb}: XOR {sum(xor)}, OR {sum(orc)}, "
          f"AND {sum(andc)} -- all conserved.\n")

    # verify against the O(n^2) definition
    assert xor == brute_convolution(a, b, lambda i, j: i ^ j)
    assert orc == brute_convolution(a, b, lambda i, j: i | j)
    assert andc == brute_convolution(a, b, lambda i, j: i & j)
    print("  All three match the brute-force O(n^2) definition exactly (integer, no rounding).")

    print("\n  The FWHT computes XOR convolution in O(n log n) by transforming both inputs with the")
    print("  Hadamard butterfly (x,y)->(x+y,x-y), multiplying pointwise, and inverse-transforming --")
    print("  the same convolution theorem that makes the FFT fast, but for the XOR group. OR and AND")
    print("  convolutions use the subset-sum (zeta) and superset-sum transforms over the bit lattice.")

    _svg(os.path.join(outdir, "walsh_hadamard.svg"), a, b, xor, orc, andc)
    print(f"\n  wrote {os.path.join(outdir, 'walsh_hadamard.svg')}")


def _svg(path, a, b, xor, orc, andc, width=760, height=470):
    n = len(a)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Combining two 3-bit distributions by XOR, OR, and AND</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'each panel: the weight on every value 0..7 after convolving A with B under one operation'
        f'</text>',
    ]

    panels = [
        ("A", a, "#4dabf7"),
        ("B", b, "#8b949e"),
        ("A XOR B", xor, "#06d6a0"),
        ("A OR B", orc, "#ffd43b"),
        ("A AND B", andc, "#ff922b"),
    ]
    px = 60
    pw = 128
    gap = 12
    base = 400
    top = 90
    for pi, (label, data, col) in enumerate(panels):
        ox = px + pi * (pw + gap)
        mx = max(data) if max(data) > 0 else 1
        bw = pw / n
        for k, v in enumerate(data):
            bh = (base - top) * v / mx
            x = ox + k * bw
            y = base - bh
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-1.5:.1f}" '
                         f'height="{bh:.1f}" fill="{col}"/>')
        parts.append(f'<line x1="{ox}" y1="{base}" x2="{ox+pw}" y2="{base}" '
                     f'stroke="#30363d" stroke-width="1"/>')
        parts.append(f'<text x="{ox+pw/2:.0f}" y="{base+22}" fill="#e6edf3" font-size="12" '
                     f'text-anchor="middle">{label}</text>')
        parts.append(f'<text x="{ox+pw/2:.0f}" y="{base+38}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">sum {sum(data)}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
