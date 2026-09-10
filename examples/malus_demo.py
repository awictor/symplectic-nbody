"""Demo: Malus's law -- dialling light down with polarizers.

Prints the Malus cos^2 transmission versus angle, the crossed/three-polarizer results, and
the quantum-Zeno-like rotating stack, then draws two curves: the cos^2 law, and the three-
polarizer rescue as the middle polarizer angle is swept.

    python examples/malus_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from malus import (malus_transmission, two_polarizer_transmission,  # noqa: E402
                   stack_transmission, three_polarizer_rescue)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Malus's law I = I0 cos^2(theta): a polarizer passes the aligned field component\n")
    print(f"  {'angle':>8}{'transmission':>15}")
    for deg in (0, 30, 45, 60, 90):
        print(f"  {deg:>5} deg{malus_transmission(1.0, math.radians(deg)):>14.3f}")

    print("\n  Crossed polarizers (90 deg apart) pass nothing: %.3f"
          % two_polarizer_transmission(1.0, math.radians(90.0)))
    print("  ...but a third polarizer at 45 deg between them rescues %.3f of the input"
          % three_polarizer_rescue(1.0, math.radians(45.0)))
    print("  (I0/8) -- light reappears where two polarizers alone gave darkness.\n")

    print("  Rotating a stack of N polarizers through 90 deg (quantum-Zeno-like):")
    for N in (1, 2, 5, 20, 100):
        print(f"    N = {N:>3}  ->  throughput {stack_transmission(1.0, math.radians(90.0), N):.3f}")
    print("  With one polarizer a 90 deg turn kills the light; with many tiny steps it")
    print("  survives -- each cos^2 is nearly 1, so the polarization is dragged around intact.")

    _svg(os.path.join(outdir, "malus.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'malus.svg')}")


def _svg(path, size=720, pad=72):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Malus&#39;s law and the three-polarizer trick</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'I = I0 cos^2(theta) (top); light rescued by a middle polarizer between crossed pair (bottom)</text>',
    ]

    x0, x1 = pad, size - pad
    mid = size * 0.52

    # --- top: cos^2 curve ---
    ty0, ty1 = mid - 26, pad + 44
    def TX(deg):
        return x0 + deg / 180.0 * (x1 - x0)
    def TY(I):
        return ty0 - I * (ty0 - ty1)
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x1}" y2="{ty0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x0}" y2="{ty1}" stroke="#8b949e" stroke-width="1.4"/>')
    for deg in range(0, 181, 45):
        parts.append(f'<text x="{TX(deg):.1f}" y="{ty0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{deg}</text>')
    for I in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{x0-6:.1f}" y="{TY(I)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{I:.1f}</text>')
    pts = []
    for i in range(181):
        pts.append(f"{TX(i):.1f},{TY(malus_transmission(1.0, math.radians(i))):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')
    # mark 45 (half) and 90 (zero)
    parts.append(f'<circle cx="{TX(45):.1f}" cy="{TY(0.5):.1f}" r="4" fill="#ffd43b"/>')
    parts.append(f'<text x="{TX(45)+6:.1f}" y="{TY(0.5)-6:.1f}" fill="#ffd43b" font-size="10">45 deg: half</text>')
    parts.append(f'<circle cx="{TX(90):.1f}" cy="{TY(0.0):.1f}" r="4" fill="#ff6b6b"/>')
    parts.append(f'<text x="{TX(90)+6:.1f}" y="{TY(0.0)-6:.1f}" fill="#ff6b6b" font-size="10">90 deg: dark</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{ty0+28:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">polarizer angle theta (deg)</text>')

    # --- bottom: three-polarizer rescue vs middle angle ---
    by0, by1 = size - pad, mid + 40
    def BX(deg):
        return x0 + deg / 90.0 * (x1 - x0)
    def BY(I):
        return by0 - I / 0.125 * (by0 - by1)     # peak is I0/8 = 0.125
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x1}" y2="{by0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x0}" y2="{by1}" stroke="#8b949e" stroke-width="1.4"/>')
    for deg in range(0, 91, 15):
        parts.append(f'<text x="{BX(deg):.1f}" y="{by0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{deg}</text>')
    rpts = []
    for i in range(91):
        rpts.append(f"{BX(i):.1f},{BY(three_polarizer_rescue(1.0, math.radians(i))):.1f}")
    parts.append(f'<polyline points="{" ".join(rpts)}" fill="none" stroke="#06d6a0" stroke-width="2.6"/>')
    parts.append(f'<circle cx="{BX(45):.1f}" cy="{BY(0.125):.1f}" r="4" fill="#ffd43b"/>')
    parts.append(f'<text x="{BX(45)+6:.1f}" y="{BY(0.125)-6:.1f}" fill="#ffd43b" font-size="10">'
                 f'peak I0/8 at 45 deg</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{by0+28:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">middle-polarizer angle (deg), outer pair crossed at 0 &amp; 90</text>')
    parts.append(f'<text x="{x0+6:.1f}" y="{by1-6:.1f}" fill="#8b949e" font-size="10">'
                 f'transmitted intensity (crossed outer pair alone = 0)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
