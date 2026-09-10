"""Demo: the quantum Hall effect -- resistance quantized to constants.

Prints the von Klitzing constant and the first few quantized Hall plateaus, the Landau-level
spacing and degeneracy versus field, then draws the Hall resistance staircase: flat plateaus
at R_K/nu as the field sweeps a fixed electron density through integer filling factors.

    python examples/quantum_hall_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantum_hall import (von_klitzing_constant, hall_resistance,  # noqa: E402
                          landau_level_spacing, landau_degeneracy, filling_factor)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Quantum Hall effect: R_xy = R_K / nu, R_K = h/e^2 = %.3f ohm\n"
          % von_klitzing_constant())
    print(f"  {'filling nu':>12}{'Hall resistance':>18}")
    for nu in (1, 2, 3, 4, 6):
        print(f"  {nu:>12}{hall_resistance(nu):>13.1f} ohm")

    print("\n  Landau levels vs field (need k_B T << spacing to resolve plateaus):")
    print(f"  {'B (T)':>8}{'spacing (meV)':>16}{'degeneracy (1/m^2)':>22}")
    for B in (2, 5, 10, 20):
        print(f"  {B:>8}{landau_level_spacing(B)/1.602e-19*1000:>13.2f}{landau_degeneracy(B):>22.2e}")

    print("\n  A 2D electron gas at n = 5e15 /m^2 hits integer filling as field is swept:")
    for B in (5, 10, 20):
        nu = filling_factor(5e15, B)
        print(f"    B = {B:>2} T  ->  nu = {nu:.2f}")

    print("\n  On each plateau the bulk is insulating and only chiral edge channels conduct,")
    print("  one per filled Landau level -- so R_xy depends on nothing but h/e^2. Reproducible")
    print("  to parts per billion in any device, it is how the ohm is now defined.")

    _svg(os.path.join(outdir, "quantum_hall.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'quantum_hall.svg')}")


def _svg(path, size=720, pad=80):
    # R_xy vs field for fixed density n: R_xy = R_K / round(nu), plateaus where nu is integer.
    n = 5e15
    B_min, B_max = 2.0, 30.0
    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 44
    R_K = von_klitzing_constant()
    r_max = R_K / 1 * 1.1     # nu=1 is the highest plateau in range

    def X(B):
        return x0 + (B - B_min) / (B_max - B_min) * (x1 - x0)

    def Y(R):
        return y0 - R / r_max * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The quantum Hall staircase</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'Hall resistance locks onto flat plateaus at R_K/nu -- values set by h/e^2 alone</text>',
    ]

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.5"/>')

    # plateau gridlines for nu = 1..6
    for nu in range(1, 7):
        R = R_K / nu
        gy = Y(R)
        if y1 <= gy <= y0:
            parts.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" '
                         f'stroke="#21262d" stroke-width="1"/>')
            parts.append(f'<text x="{x0-6:.1f}" y="{gy+3:.1f}" fill="#8b949e" font-size="9" '
                         f'text-anchor="end">R_K/{nu}</text>')

    # the staircase: sweep B, R = R_K / round(filling factor) (plateau to nearest integer nu)
    pts = []
    n_pts = 400
    prev_nu = None
    for i in range(n_pts + 1):
        B = B_min + (B_max - B_min) * i / n_pts
        nu = filling_factor(n, B)
        nu_int = max(1, round(nu))
        R = R_K / nu_int
        pts.append(f"{X(B):.1f},{Y(R):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')

    # label a couple of plateaus with their nu
    for nu in (2, 3, 4):
        # field where filling = nu: B = n h / (e nu)
        B_nu = n * 6.626e-34 / (1.602e-19 * nu)
        if B_min <= B_nu <= B_max:
            parts.append(f'<circle cx="{X(B_nu):.1f}" cy="{Y(R_K/nu):.1f}" r="4" fill="#ffd43b"/>')
            parts.append(f'<text x="{X(B_nu):.1f}" y="{Y(R_K/nu)-8:.1f}" fill="#ffd43b" '
                         f'font-size="10" text-anchor="middle">nu={nu}</text>')

    for B in range(5, 31, 5):
        parts.append(f'<text x="{X(B):.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{B} T</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+32:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">magnetic field (n = 5e15 /m^2 fixed)</text>')
    parts.append(f'<text x="24" y="{(y0+y1)/2:.1f}" fill="#8b949e" font-size="11" '
                 f'transform="rotate(-90 24 {(y0+y1)/2:.1f})" text-anchor="middle">Hall resistance R_xy</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
