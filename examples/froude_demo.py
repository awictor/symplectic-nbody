"""Demo: the Froude number -- subcritical, supercritical, and the hydraulic jump.

Prints flow regimes across speed and depth, the hull speed of several boats, and the Kelvin
wake angle, then draws a hydraulic jump in profile: a thin sheet of fast supercritical water
slamming into a deep, slow subcritical pool, with the conjugate depths and Froude numbers
labelled.

    python examples/froude_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from froude import (wave_speed, froude_number, hull_froude, flow_regime,  # noqa: E402
                    hull_speed, conjugate_depth, kelvin_wake_half_angle)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Froude number Fr = U / sqrt(g h): flow speed vs its own wave speed\n")
    print(f"  {'flow':<30}{'U (m/s)':>9}{'h (m)':>8}{'Fr':>7}{'regime':>16}")
    flows = [
        ("lazy river", 0.5, 2.0),
        ("brisk stream", 1.5, 0.5),
        ("below a spillway", 6.0, 0.2),
        ("kitchen-sink disc", 1.0, 0.001),
    ]
    for label, U, h in flows:
        Fr = froude_number(U, h)
        print(f"  {label:<30}{U:>9.2f}{h:>8.3f}{Fr:>7.2f}{flow_regime(Fr):>16}")

    print("\n  Hull speed (displacement wall at Fr ~ 0.40):")
    print(f"  {'boat / waterline':<28}{'L (m)':>8}{'V_hull (m/s)':>14}{'knots':>9}")
    for label, L in (("kayak", 4.0), ("day-sailer", 7.0), ("yacht", 12.0), ("clipper", 60.0)):
        V = hull_speed(L)
        print(f"  {label:<28}{L:>8.1f}{V:>14.2f}{V*1.94384:>9.1f}")

    print("\n  Kelvin ship-wake half-angle: %.2f deg -- fixed, whatever the speed.\n"
          % kelvin_wake_half_angle())
    h2 = conjugate_depth(0.2, 3.0)
    print("  A hydraulic jump: supercritical Fr1=3 water 0.2 m deep leaps to %.2f m and goes" % h2)
    print("  subcritical, dumping its excess energy into turbulence -- the step below a weir.")

    _svg(os.path.join(outdir, "froude.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'froude.svg')}")


def _svg(path, size=720, pad=70):
    # Hydraulic jump profile: upstream thin/fast (Fr1>1), downstream deep/slow (Fr2<1).
    h1, Fr1 = 0.2, 3.0
    h2 = conjugate_depth(h1, Fr1)
    U1 = Fr1 * wave_speed(h1)
    q = U1 * h1
    U2 = q / h2
    Fr2 = froude_number(U2, h2)

    x0, x1 = pad, size - pad
    floor = size * 0.80
    scale = (floor - size * 0.28) / h2        # px per metre so h2 fits

    xj = (x0 + x1) * 0.46                      # jump location

    def Yd(h):
        return floor - h * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The hydraulic jump</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'fast shallow (supercritical) water leaps to a deep slow (subcritical) pool below a weir</text>',
    ]

    # channel floor
    parts.append(f'<line x1="{x0-10:.1f}" y1="{floor:.1f}" x2="{x1+10:.1f}" y2="{floor:.1f}" '
                 f'stroke="#8b949e" stroke-width="2"/>')

    # upstream thin fast sheet
    parts.append(f'<rect x="{x0:.1f}" y="{Yd(h1):.1f}" width="{xj-x0:.1f}" height="{h1*scale:.1f}" '
                 f'fill="#4dabf7" opacity="0.55"/>')
    # the jump face (rising surface) + downstream deep pool
    parts.append(f'<polygon points="{xj:.1f},{Yd(h1):.1f} {xj+40:.1f},{Yd(h2):.1f} '
                 f'{x1:.1f},{Yd(h2):.1f} {x1:.1f},{floor:.1f} {xj:.1f},{floor:.1f}" '
                 f'fill="#4dabf7" opacity="0.55"/>')
    # turbulent roller at the jump
    parts.append(f'<circle cx="{xj+18:.1f}" cy="{Yd(h2)+12:.1f}" r="12" fill="none" '
                 f'stroke="#e6edf3" stroke-width="1.5" opacity="0.6"/>')
    parts.append(f'<circle cx="{xj+34:.1f}" cy="{Yd(h2)+22:.1f}" r="8" fill="none" '
                 f'stroke="#e6edf3" stroke-width="1.2" opacity="0.5"/>')

    # flow arrows
    parts.append(f'<line x1="{x0+30:.1f}" y1="{Yd(h1/2):.1f}" x2="{xj-30:.1f}" y2="{Yd(h1/2):.1f}" '
                 f'stroke="#ffd43b" stroke-width="2.5"/>')
    parts.append(f'<polygon points="{xj-30:.1f},{Yd(h1/2):.1f} {xj-42:.1f},{Yd(h1/2)-5:.1f} '
                 f'{xj-42:.1f},{Yd(h1/2)+5:.1f}" fill="#ffd43b"/>')

    # labels
    parts.append(f'<text x="{(x0+xj)/2:.1f}" y="{Yd(h1)-10:.1f}" fill="#4dabf7" font-size="12" '
                 f'text-anchor="middle">Fr1 = {Fr1:.1f} (supercritical)</text>')
    parts.append(f'<text x="{(x0+xj)/2:.1f}" y="{floor+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">h1 = {h1:.2f} m, U1 = {U1:.1f} m/s</text>')
    parts.append(f'<text x="{(xj+x1)/2:.1f}" y="{Yd(h2)-10:.1f}" fill="#06d6a0" font-size="12" '
                 f'text-anchor="middle">Fr2 = {Fr2:.2f} (subcritical)</text>')
    parts.append(f'<text x="{(xj+x1)/2:.1f}" y="{floor+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">h2 = {h2:.2f} m, U2 = {U2:.1f} m/s</text>')

    # depth scale bar
    parts.append(f'<text x="{x0-6:.1f}" y="{Yd(h2)+4:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end">{h2:.2f} m</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
