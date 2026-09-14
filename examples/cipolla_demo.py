"""Cipolla demo: modular square roots via a quadratic field, contrasted with Tonelli-Shanks (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cipolla as CP
import tonelli_shanks as TS


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
RED = "#ff6b6b"
YELLOW = "#ffd43b"


def main(outdir=None):
    p = 37
    lines = []
    lines.append("Cipolla's algorithm: modular square roots via GF(p^2)")
    lines.append("=" * 56)
    lines.append(f"solve x^2 = n (mod {p}) by working in the field GF({p}^2)")
    lines.append("pick a with a^2 - n a non-residue, then x = (a + w)^((p+1)/2), w^2 = a^2 - n")
    lines.append("")
    lines.append(f"{'n':>4}{'QR?':>6}{'Cipolla roots':>18}{'Tonelli':>10}")
    for n in range(1, 16):
        qr = CP.is_quadratic_residue(n, p)
        roots = CP.both_roots(n, p)
        t = TS.sqrt_mod(n, p)
        rstr = f"{roots}" if roots else "-- (none)"
        lines.append(f"{n:>4}{('yes' if qr else 'no'):>6}{rstr:>18}{str(t):>10}")
    lines.append("")
    lines.append("Both algorithms return valid roots (possibly differing by sign). Cipolla needs")
    lines.append("no case analysis on the powers of 2 dividing p-1, so it shines when that power")
    lines.append("is large -- one exponentiation in a 2-dimensional field and the root drops out.")
    lines.append("")
    # a big prime demonstration
    P = 1000000007
    n = 2
    r = CP.sqrt_mod(n, P)
    lines.append(f"big prime: sqrt(2) mod {P} = {r}")
    lines.append(f"  check {r}^2 mod {P} = {(r*r) % P} == 2? {(r*r) % P == 2}")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # SVG: plot x -> x^2 mod p as points, highlight the two preimages of a chosen n
        W, H = 640, 460
        ml, mt, side = 60, 60, 360
        target = 10                                  # highlight sqrt(target)
        roots = CP.both_roots(target, p) or ()

        def sx(x):
            return ml + x / (p - 1) * side

        def sy(y):
            return mt + side - y / (p - 1) * side

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'x -&gt; x^2 mod {p}; Cipolla inverts it (roots of {target} in red)</text>')
        s.append(f'<rect x="{ml}" y="{mt}" width="{side}" height="{side}" fill="none" '
                 f'stroke="{GRAY}" stroke-width="0.6"/>')
        # horizontal line at y = target
        s.append(f'<line x1="{ml}" y1="{sy(target):.1f}" x2="{ml+side}" y2="{sy(target):.1f}" '
                 f'stroke="{YELLOW}" stroke-width="1" stroke-dasharray="4,3"/>')
        s.append(f'<text x="{ml+side+4}" y="{sy(target)+4:.1f}" fill="{YELLOW}" '
                 f'font-size="10">n={target}</text>')
        for x in range(p):
            y = (x * x) % p
            col = RED if x in roots else BLUE
            r_pt = 4 if x in roots else 2.5
            s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{r_pt}" fill="{col}"/>')
        s.append(f'<text x="{ml}" y="{mt+side+30}" fill="{GRAY}" font-size="11">x (input)</text>')
        s.append(f'<text x="{ml-45}" y="{mt+side/2:.0f}" fill="{GRAY}" font-size="11">x^2 mod p</text>')
        s.append(f'<text x="{ml}" y="{H-14}" fill="{GRAY}" font-size="10">'
                 f'Squaring is 2-to-1 on the residues; the two red points on the dashed line are the '
                 f'square roots Cipolla recovers with one field exponentiation.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "cipolla.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
