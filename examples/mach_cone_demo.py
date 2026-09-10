"""Demo: the Mach cone -- the geometry of supersonic flight.

Prints the Mach-cone angle and sonic-boom timing across speeds, the Prandtl-Glauert lift
rise below Mach 1, and the Prandtl-Meyer angle above it, then draws the Mach cone: a
supersonic source with its expanding wavelets and the trailing cone envelope of half-angle
arcsin(1/M).

    python examples/mach_cone_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mach_cone import (mach_angle, mach_angle_deg, sonic_boom_ground_offset,  # noqa: E402
                       sonic_boom_delay, prandtl_glauert_factor, prandtl_meyer_angle)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Mach cone: sin(mu) = 1/M -- faster flight, tighter cone\n")
    print(f"  {'Mach':>6}{'cone half-angle':>18}{'boom lag @ 12 km':>20}")
    for M in (1.0, 1.2, 2.0, 3.0, 5.0):
        mu = mach_angle_deg(M)
        lag = sonic_boom_delay(12000.0, M, 295.0)     # c ~ 295 m/s at altitude
        print(f"  {M:>6.1f}{mu:>15.2f} deg{lag:>17.1f} s")

    print("\n  Subsonic compressibility (Prandtl-Glauert lift factor 1/sqrt(1-M^2)):")
    for M in (0.3, 0.6, 0.8, 0.9, 0.95):
        print(f"    M = {M:.2f}  ->  x{prandtl_glauert_factor(M):.2f}")

    print("\n  Supersonic turning (Prandtl-Meyer angle, gamma=1.4):")
    for M in (1.5, 2.0, 3.0, 5.0):
        print(f"    M = {M:.1f}  ->  nu = {math.degrees(prandtl_meyer_angle(M)):.1f} deg")

    off = sonic_boom_ground_offset(12000.0, 2.0)
    print("\n  At Mach 2 and 12 km, the boom carpet lands %.1f km behind the overhead point," % (off/1000))
    print("  reaching the ground seconds after the jet has already gone by -- and because the")
    print("  cone trails at a fixed angle, it sweeps a continuous 'boom carpet' along the track.")

    _svg(os.path.join(outdir, "mach_cone.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'mach_cone.svg')}")


def _svg(path, size=720, pad=60):
    # Source flying left->right at Mach M; draw wavelets emitted at past times and the cone.
    M = 2.0
    x0, x1 = pad, size - pad
    cy = size * 0.40
    # source now at xs; it has travelled from the left. Wavelets emitted at fractions back.
    xs = x1 - 40
    U_px = (x1 - x0 - 80)          # pixels the source has covered in the window
    # wavelet emitted at time fraction f ago was at xs - f*U_px, radius = f*U_px/M
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The Mach cone at M = 2</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'each wavelet expands at the sound speed; the source outruns them, leaving a cone of half-angle arcsin(1/M)</text>',
    ]

    # flight path
    parts.append(f'<line x1="{x0-10:.1f}" y1="{cy:.1f}" x2="{x1+10:.1f}" y2="{cy:.1f}" '
                 f'stroke="#21262d" stroke-width="1"/>')

    # wavelets
    for f in (0.25, 0.5, 0.75, 1.0):
        xc = xs - f * U_px
        r = f * U_px / M
        parts.append(f'<circle cx="{xc:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" '
                     f'stroke="#4dabf7" stroke-width="1.6" opacity="0.7"/>')

    # the Mach cone: tangent lines from the source back over the wavelets
    mu = mach_angle(M)
    L = U_px * 1.05
    dx = L * math.cos(mu)
    dy = L * math.sin(mu)
    parts.append(f'<line x1="{xs:.1f}" y1="{cy:.1f}" x2="{xs-dx:.1f}" y2="{cy-dy:.1f}" '
                 f'stroke="#ff6b6b" stroke-width="2.4"/>')
    parts.append(f'<line x1="{xs:.1f}" y1="{cy:.1f}" x2="{xs-dx:.1f}" y2="{cy+dy:.1f}" '
                 f'stroke="#ff6b6b" stroke-width="2.4"/>')

    # angle arc + label
    parts.append(f'<path d="M {xs-60:.1f} {cy:.1f} A 60 60 0 0 0 {xs-60*math.cos(mu):.1f} '
                 f'{cy-60*math.sin(mu):.1f}" fill="none" stroke="#ffd43b" stroke-width="1.5"/>')
    parts.append(f'<text x="{xs-92:.1f}" y="{cy-20:.1f}" fill="#ffd43b" font-size="13">'
                 f'mu = {mach_angle_deg(M):.0f} deg</text>')

    # source (aircraft) marker + velocity arrow
    parts.append(f'<polygon points="{xs+12:.1f},{cy:.1f} {xs-8:.1f},{cy-6:.1f} {xs-8:.1f},{cy+6:.1f}" '
                 f'fill="#e6edf3"/>')
    parts.append(f'<text x="{xs-4:.1f}" y="{cy-14:.1f}" fill="#e6edf3" font-size="11" '
                 f'text-anchor="end">source at Mach 2</text>')

    # ground line + boom carpet note
    ground = size * 0.85
    parts.append(f'<line x1="{x0-10:.1f}" y1="{ground:.1f}" x2="{x1+10:.1f}" y2="{ground:.1f}" '
                 f'stroke="#8b949e" stroke-width="2"/>')
    parts.append(f'<text x="{x0:.1f}" y="{ground+20:.1f}" fill="#8b949e" font-size="11">'
                 f'the cone sweeps the ground as a continuous sonic-boom carpet along the flight track</text>')
    # where cone meets ground
    x_ground = xs - (ground - cy) / math.tan(mu)
    parts.append(f'<circle cx="{x_ground:.1f}" cy="{ground:.1f}" r="5" fill="#ff6b6b"/>')
    parts.append(f'<line x1="{xs:.1f}" y1="{cy:.1f}" x2="{x_ground:.1f}" y2="{ground:.1f}" '
                 f'stroke="#ff6b6b" stroke-width="1" stroke-dasharray="4 4" opacity="0.6"/>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
