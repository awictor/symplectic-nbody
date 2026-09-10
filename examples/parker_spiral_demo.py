"""Demo: the Parker spiral of the interplanetary magnetic field.

Prints the garden-hose angle and field strength from the Sun out to Jupiter, then
draws several field lines spiralling out through the ecliptic -- the rotating-
sprinkler pattern the Sun's rotation stamps on the radially-flowing solar wind.

    python examples/parker_spiral_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parker_spiral import (spiral_angle, phi_of_r, field_magnitude,  # noqa: E402
                           field_components, AU, R_SUN, OMEGA_SUN)

U = 4e5  # 400 km/s solar wind


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("The Parker spiral: the Sun's field wound up by its rotation\n")
    print(f"  solar wind speed: {U/1e3:.0f} km/s, "
          f"rotation period: {2*math.pi/OMEGA_SUN/86400:.1f} days\n")
    print(f"  {'location':>14}{'r (AU)':>9}{'angle (deg)':>13}{'|B| (nT)':>11}")
    print("  " + "-" * 47)
    places = [("Mercury", 0.39), ("Venus", 0.72), ("Earth", 1.0),
              ("Mars", 1.52), ("Jupiter", 5.20), ("Saturn", 9.58)]
    for name, r_au in places:
        r = r_au * AU
        psi = math.degrees(spiral_angle(r, U))
        B = field_magnitude(r, U, 5e-9) * 1e9
        print(f"  {name:>14}{r_au:>9.2f}{psi:>13.1f}{B:>11.2f}")

    print("\n  Near the Sun the field is nearly radial; by Earth it is bent ~45 deg")
    print("  (the classic garden-hose angle); past Jupiter it is nearly azimuthal.")
    print("  B_r falls as 1/r^2 but B_phi only as 1/r, so the distant heliospheric")
    print("  field is mostly the wound-up azimuthal component. This is why western-")
    print("  limb solar flares connect best to Earth along the spiral field line.")

    _svg(os.path.join(outdir, "parker_spiral.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'parker_spiral.svg')}")


def _svg(path, size=640, rmax_au=6.0):
    cx = cy = size / 2.0
    scale = (size / 2.0 - 40) / rmax_au

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # planet orbit circles
    for r_au, label in [(1.0, "Earth"), (5.2, "Jupiter")]:
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r_au*scale:.1f}" '
                     f'fill="none" stroke="#30363d" stroke-width="1"/>')
        parts.append(f'<text x="{cx + r_au*scale + 4:.1f}" y="{cy:.1f}" '
                     f'fill="#484f58" font-size="10">{label}</text>')
    # the Sun
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="#ffd43b"/>')

    # several field lines from footpoints spaced 90 deg apart, two polarities
    colors = ["#8338ec", "#3a86ff", "#ff6b6b", "#06d6a0"]
    steps = 400
    for k in range(4):
        phi0 = k * math.pi / 2.0
        pts = []
        for i in range(steps + 1):
            r = R_SUN + (rmax_au * AU - R_SUN) * i / steps
            phi = phi_of_r(r, U, phi0=phi0)
            rr = r / AU * scale
            x = cx + rr * math.cos(phi)
            y = cy - rr * math.sin(phi)
            pts.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                     f'stroke="{colors[k]}" stroke-width="1.8"/>')

    parts.append(f'<text x="20" y="30" fill="#e6edf3" font-size="17">'
                 f'The Parker spiral (ecliptic plane)</text>')
    parts.append(f'<text x="20" y="50" fill="#8b949e" font-size="11">'
                 f'radial wind + solar rotation = Archimedean spiral field lines</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
