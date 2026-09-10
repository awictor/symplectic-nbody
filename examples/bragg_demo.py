"""Demo: Bragg diffraction -- reading a crystal with X-rays.

Prints the Bragg angles for successive orders and Miller planes of a silicon crystal, then
draws the Bragg geometry: X-rays reflecting off two atomic planes with the extra path length
2 d sin(theta) that must equal a whole number of wavelengths.

    python examples/bragg_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bragg import (bragg_angle, bragg_angle_deg, cubic_spacing, max_order)  # noqa: E402


LAMBDA = 0.1541e-9      # Cu K-alpha X-ray
A_SI = 0.543e-9         # silicon lattice constant


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Bragg's law n lambda = 2 d sin(theta): X-ray crystallography\n")
    print("  Cu K-alpha (0.1541 nm) on silicon (a = 0.543 nm)\n")
    print(f"  {'(hkl)':>8}{'d (nm)':>10}{'1st-order angle':>18}{'max order':>11}")
    for hkl in ((1, 1, 1), (2, 2, 0), (3, 1, 1), (4, 0, 0)):
        d = cubic_spacing(A_SI, *hkl)
        th = bragg_angle_deg(LAMBDA, d, 1)
        print(f"  {str(hkl):>8}{d*1e9:>10.4f}{th:>15.2f} deg{max_order(LAMBDA, d):>11}")

    d111 = cubic_spacing(A_SI, 1, 1, 1)
    print("\n  Orders off the (111) planes (d = %.4f nm):" % (d111 * 1e9))
    n = 1
    while True:
        try:
            print(f"    n = {n}  ->  theta = {bragg_angle_deg(LAMBDA, d111, n):.2f} deg")
        except ValueError:
            break
        n += 1

    print("\n  Each family of planes flashes a reflection only where its path difference")
    print("  2 d sin(theta) is a whole number of wavelengths. Reading the spots backwards gives")
    print("  the atomic spacings -- how crystallography solved salt, DNA, and countless proteins.")

    _svg(os.path.join(outdir, "bragg.svg"), d111)
    print(f"\n  wrote {os.path.join(outdir, 'bragg.svg')}")


def _svg(path, d_plane, size=720, pad=70):
    theta = bragg_angle(LAMBDA, d_plane, 1)     # first-order angle
    x0, x1 = pad, size - pad

    # two atomic planes, spacing shown; incident + reflected rays at glancing angle theta
    plane_top = size * 0.40
    plane_gap = size * 0.20
    plane_bot = plane_top + plane_gap

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Bragg reflection from atomic planes</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'the lower ray travels an extra 2 d sin(theta); reflections flash when that equals n lambda</text>',
    ]

    # draw two planes as rows of atoms
    for py in (plane_top, plane_bot):
        parts.append(f'<line x1="{x0:.1f}" y1="{py:.1f}" x2="{x1:.1f}" y2="{py:.1f}" '
                     f'stroke="#30363d" stroke-width="1.5"/>')
        for i in range(13):
            ax = x0 + (x1 - x0) * i / 12
            parts.append(f'<circle cx="{ax:.1f}" cy="{py:.1f}" r="5" fill="#8b949e"/>')

    # spacing label
    mx = x0 + 40
    parts.append(f'<line x1="{mx:.1f}" y1="{plane_top:.1f}" x2="{mx:.1f}" y2="{plane_bot:.1f}" '
                 f'stroke="#ffd43b" stroke-width="1.2"/>')
    parts.append(f'<text x="{mx+6:.1f}" y="{(plane_top+plane_bot)/2:.1f}" fill="#ffd43b" '
                 f'font-size="12">d = {d_plane*1e9:.3f} nm</text>')

    # geometry: rays hit the middle of the planes at glancing angle theta
    hitx = (x0 + x1) / 2
    dx = 180
    dy = dx * math.tan(theta)

    # incident + reflected on the TOP plane
    parts.append(f'<line x1="{hitx-dx:.1f}" y1="{plane_top-dy:.1f}" x2="{hitx:.1f}" y2="{plane_top:.1f}" '
                 f'stroke="#4dabf7" stroke-width="2.2"/>')
    parts.append(f'<line x1="{hitx:.1f}" y1="{plane_top:.1f}" x2="{hitx+dx:.1f}" y2="{plane_top-dy:.1f}" '
                 f'stroke="#4dabf7" stroke-width="2.2"/>')
    # incident + reflected on the BOTTOM plane (offset to same entry direction)
    parts.append(f'<line x1="{hitx-dx:.1f}" y1="{plane_bot-dy:.1f}" x2="{hitx:.1f}" y2="{plane_bot:.1f}" '
                 f'stroke="#06d6a0" stroke-width="2.2"/>')
    parts.append(f'<line x1="{hitx:.1f}" y1="{plane_bot:.1f}" x2="{hitx+dx:.1f}" y2="{plane_bot-dy:.1f}" '
                 f'stroke="#06d6a0" stroke-width="2.2"/>')

    # angle marker at the top hit
    parts.append(f'<path d="M {hitx-50:.1f} {plane_top:.1f} A 50 50 0 0 0 '
                 f'{hitx-50*math.cos(theta):.1f} {plane_top-50*math.sin(theta):.1f}" '
                 f'fill="none" stroke="#ff922b" stroke-width="1.4"/>')
    parts.append(f'<text x="{hitx-64:.1f}" y="{plane_top-14:.1f}" fill="#ff922b" font-size="12">'
                 f'theta = {math.degrees(theta):.1f} deg</text>')

    parts.append(f'<text x="{hitx-dx:.1f}" y="{plane_top-dy-8:.1f}" fill="#4dabf7" font-size="11">'
                 f'incident X-rays</text>')
    parts.append(f'<text x="{hitx+dx-70:.1f}" y="{plane_top-dy-8:.1f}" fill="#4dabf7" font-size="11">'
                 f'reflected (in phase)</text>')

    parts.append(f'<text x="{x0:.1f}" y="{size-30:.1f}" fill="#8b949e" font-size="11">'
                 f'n lambda = 2 d sin(theta) -- constructive interference builds the diffracted beam</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
