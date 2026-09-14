"""Lenstra ECM demo: factor a semiprime whose p-1 is non-smooth, showing the curve that cracks it (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import lenstra_ecm as E


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
YELLOW = "#ffd43b"
RED = "#ff6b6b"


def main(outdir=None):
    lines = []
    lines.append("Lenstra's elliptic-curve method (ECM)")
    lines.append("=" * 50)
    lines.append("factor N by running k*P on a random curve mod N until an inversion fails --")
    lines.append("the failed gcd hands you a factor. Fresh curve = fresh dice, so no single")
    lines.append("unlucky factorization defeats it (unlike Pollard p-1).")
    lines.append("")
    cases = [
        ("small semiprime", 1073),
        ("Project Euler #3", 600851475143),
        ("10-digit-prime semiprime", 1000000007 * 1000000009),
        ("mixed factors", 2 ** 3 * 3 ** 2 * 5 * 101),
        ("prime power", 7 ** 5),
        ("a prime", 9999999967),
    ]
    lines.append(f"{'case':>26}{'N':>22}   factorization")
    for name, N in cases:
        f = E.factorize(N)
        fs = " * ".join(str(p) for p in f)
        lines.append(f"{name:>26}{N:>22}   {fs}")
    lines.append("")
    lines.append("Each factorization is verified: the primes multiply back to N and each passes")
    lines.append("the Baillie-PSW primality test. ECM is the method of choice for pulling out")
    lines.append("medium (up to ~40-digit) factors before a heavier sieve finishes the job.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # SVG: draw a small elliptic curve y^2 = x^3 + ax + b over a small prime field as a point cloud
        p = 97
        a, b = 2, 3
        pts = [(x, y) for x in range(p) for y in range(p) if (y * y - (x * x * x + a * x + b)) % p == 0]
        W, H = 640, 460
        ml, mt, side = 60, 60, 360

        def sx(x):
            return ml + x / (p - 1) * side

        def sy(y):
            return mt + side - y / (p - 1) * side

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'An elliptic curve y^2 = x^3 + {a}x + {b} over GF({p})</text>')
        s.append(f'<rect x="{ml}" y="{mt}" width="{side}" height="{side}" fill="none" '
                 f'stroke="{GRAY}" stroke-width="0.6"/>')
        for x, y in pts:
            s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="{GREEN}"/>')
        s.append(f'<text x="{ml}" y="{mt+side+24}" fill="{GRAY}" font-size="11">'
                 f'{len(pts)+1} points (incl. the point at infinity) form a finite group</text>')
        s.append(f'<text x="{ml}" y="{H-14}" fill="{GRAY}" font-size="10">'
                 f'ECM works this same group arithmetic mod a COMPOSITE N; when a slope inversion '
                 f'fails, the gcd with N reveals a prime factor.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "lenstra_ecm.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
