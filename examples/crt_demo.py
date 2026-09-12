"""Demo: the Chinese Remainder Theorem reconstructing a number from its remainders.

Solves the classic Sunzi puzzle, reconstructs a secret number from its residues modulo several primes,
shows the RSA-style speedup idea, and handles a non-coprime system. Draws the residue-grid picture:
the unique solution where all the congruence stripes align.

    python examples/crt_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from crt import crt, crt_general, extended_gcd, mod_inverse, verify  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Chinese Remainder Theorem: a number from its remainders\n")

    print("  Sunzi's classic puzzle (3rd century): a number leaves remainder")
    print("    2 mod 3, 3 mod 5, 2 mod 7 -- what is it?")
    x, M = crt([2, 3, 2], [3, 5, 7])
    print(f"    -> x = {x} (mod {M}); check: {x}%3={x%3}, {x}%5={x%5}, {x}%7={x%7}")

    # reconstruct a larger secret
    secret = 8675309
    moduli = [101, 103, 107, 109, 113]
    remainders = [secret % m for m in moduli]
    xr, Mr = crt(remainders, moduli)
    print(f"\n  Reconstruct {secret} from its residues mod {moduli}:")
    print(f"    residues: {remainders}")
    print(f"    CRT gives {xr} (mod {Mr}); matches the secret: {xr == secret}")

    # RSA-style speedup illustration
    print("\n  RSA speedup idea: instead of one exponentiation mod n = p*q,")
    print("  do two mod p and mod q (each ~half the size, ~4x faster) and recombine with CRT.")
    p, q = 1000003, 1000033
    val = 123456789 % (p * q)
    xr2, _ = crt([val % p, val % q], [p, q])
    print(f"    value {val} recovered from (mod {p}, mod {q}): {xr2 == val}")

    # non-coprime
    print("\n  Non-coprime moduli (generalized CRT):")
    res = crt_general([3, 5], [4, 6])
    print(f"    x = 3 mod 4 and x = 5 mod 6 -> {res[0]} mod {res[1]}")
    print(f"    x = 1 mod 2 and x = 0 mod 4 -> {crt_general([1,0],[2,4])} (contradiction)")

    print("\n  For pairwise-coprime moduli there's a unique answer mod their product, built as")
    print("  sum(r_i * M_i * (M_i^-1 mod m_i)) where M_i = M/m_i -- each term hits r_i in its own")
    print("  modulus and 0 in the others. The extended Euclidean algorithm supplies the inverses.")

    _svg(os.path.join(outdir, "crt.svg"), [2, 3, 2], [3, 5, 7], x)
    print(f"\n  wrote {os.path.join(outdir, 'crt.svg')}")


def _svg(path, remainders, moduli, solution, width=760, height=400):
    M = 1
    for m in moduli:
        M *= m

    m_left, m_top = 60, 90
    cell = (width - m_left - 30) / M
    row_h = 70

    colors = ["#4dabf7", "#ffd43b", "#06d6a0"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'CRT: the unique value where all congruence stripes align</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'each row shades the values matching one congruence; they coincide only at x = {solution}</text>',
    ]
    for row, (r, m) in enumerate(zip(remainders, moduli)):
        y = m_top + row * row_h
        col = colors[row % len(colors)]
        parts.append(f'<text x="20" y="{y+18:.0f}" fill="{col}" font-size="12">x%{m}={r}</text>')
        for v in range(M):
            if v % m == r:
                x = m_left + v * cell
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" height="30" '
                             f'fill="{col}" fill-opacity="0.7"/>')
    # solution marker: a vertical line through all rows
    sx = m_left + solution * cell + cell / 2
    parts.append(f'<line x1="{sx:.1f}" y1="{m_top-8:.0f}" x2="{sx:.1f}" '
                 f'y2="{m_top + len(moduli)*row_h:.0f}" stroke="#ff6b6b" stroke-width="2.5"/>')
    parts.append(f'<text x="{sx:.1f}" y="{m_top + len(moduli)*row_h + 18:.0f}" fill="#ff6b6b" '
                 f'font-size="14" text-anchor="middle">x = {solution}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
