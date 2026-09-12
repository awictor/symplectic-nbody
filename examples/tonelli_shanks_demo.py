"""Demo: Tonelli-Shanks modular square roots and quadratic residues.

Computes modular square roots for several primes, shows the residue/non-residue split via the
Legendre symbol, decompresses an elliptic-curve-style point (recovering y from x), and draws the
quadratic-residue pattern for a prime.

    python examples/tonelli_shanks_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tonelli_shanks import sqrt_mod, legendre_symbol, both_roots, is_quadratic_residue  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Tonelli-Shanks: square roots modulo a prime\n")

    print("  Modular square roots (x^2 = n mod p):")
    for n, p in [(10, 13), (2, 7), (5, 7), (123456, 1000033)]:
        br = both_roots(n, p)
        if br is None:
            print(f"    sqrt({n}) mod {p}: none ({n} is a non-residue)")
        else:
            print(f"    sqrt({n}) mod {p}: {br}  (check: {br[0]}^2 = {br[0]**2 % p} mod {p})")

    # quadratic residues of a small prime
    p = 13
    residues = [a for a in range(1, p) if is_quadratic_residue(a, p)]
    non = [a for a in range(1, p) if not is_quadratic_residue(a, p)]
    print(f"\n  Modulo {p}: {len(residues)} residues {residues}, {len(non)} non-residues {non}")
    print(f"    (exactly (p-1)/2 = {(p-1)//2} of each, per the Legendre symbol)")

    # elliptic-curve point decompression: y^2 = x^3 + ax + b (mod p)
    p = 1000003
    a, b = 2, 3
    x = 5
    rhs = (x ** 3 + a * x + b) % p
    y = sqrt_mod(rhs, p)
    print(f"\n  Elliptic-curve point decompression on y^2 = x^3 + {a}x + {b} (mod {p}):")
    if y is not None:
        print(f"    x = {x} -> y = {y} (and {p - y}); verify y^2 = {y*y % p} = rhs {rhs}: {y*y % p == rhs}")
    else:
        print(f"    x = {x}: no point (rhs {rhs} is a non-residue) -- would try another x")

    print("\n  Euler's criterion (n^((p-1)/2) mod p) tells us if a root exists; if so, Tonelli-Shanks")
    print("  finds it: for p = 3 mod 4 it's just n^((p+1)/4), otherwise it iteratively corrects a")
    print("  candidate using a quadratic non-residue until the error's order drops to zero.")

    _svg(os.path.join(outdir, "tonelli_shanks.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'tonelli_shanks.svg')}")


def _svg(path, width=760, height=430):
    # show the quadratic-residue structure of a prime as a grid, and the sqrt map
    p = 37
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Quadratic residues mod {p} (green) and the square map x -> x^2</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'top row: each value 1..{p-1} coloured as residue (green) or non-residue (gray); '
        f'curve: x^2 mod {p}</text>',
    ]

    # residue row
    residues = set((x * x) % p for x in range(1, p))
    cell = (width - 60) / (p - 1)
    for a in range(1, p):
        x = 40 + (a - 1) * cell
        col = "#06d6a0" if a in residues else "#30363d"
        parts.append(f'<rect x="{x:.1f}" y="70" width="{cell-2:.1f}" height="26" fill="{col}"/>')
        if p <= 40:
            parts.append(f'<text x="{x + cell/2:.1f}" y="88" fill="#0d1117" font-size="9" '
                         f'text-anchor="middle">{a}</text>')

    # x^2 mod p scatter
    gx0, gy0, gw, gh = 40, 130, width - 80, 240

    def px(v):
        return gx0 + v / p * gw

    def py(v):
        return gy0 + gh - v / p * gh

    parts.append(f'<line x1="{gx0}" y1="{gy0+gh}" x2="{gx0+gw}" y2="{gy0+gh}" stroke="#484f58"/>')
    parts.append(f'<line x1="{gx0}" y1="{gy0}" x2="{gx0}" y2="{gy0+gh}" stroke="#484f58"/>')
    for x in range(p):
        y = (x * x) % p
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.5" fill="#4dabf7"/>')
    parts.append(f'<text x="{gx0+gw/2:.0f}" y="{gy0+gh+20:.0f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">x (each x maps to x^2 mod {p}; two x share each residue)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
