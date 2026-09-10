"""Demo: parallax, proper motion, and space velocity.

Prints distances and space velocities for the nearest and fastest-moving stars, then
draws the parallax geometry -- Earth's orbit, the baseline, and the tiny angle a star
subtends -- to scale in angle.

    python examples/parallax_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parallax import (distance_pc, distance_ly, tangential_velocity,  # noqa: E402
                      space_velocity, parallax_at)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Parallax: d (pc) = 1 / p (arcsec);  v_t (km/s) = 4.74 mu d\n")
    print(f"  {'star':>16}{'p (\")':>9}{'d (pc)':>9}{'d (ly)':>9}"
          f"{'mu (\"/yr)':>11}{'v_space':>10}")
    print("  " + "-" * 64)
    # (name, parallax", proper motion "/yr, radial km/s)
    stars = [
        ("Proxima Cen", 0.7687, 3.85, -22.4),
        ("Barnard's Star", 0.5469, 10.36, -110.0),
        ("Sirius", 0.3792, 1.34, -5.5),
        ("Vega", 0.1305, 0.35, -13.9),
        ("Betelgeuse", 0.0055, 0.03, 21.9),
    ]
    for name, p, mu, vr in stars:
        d = distance_pc(p)
        v = space_velocity(vr, mu, d)
        print(f"  {name:>16}{p:>9.4f}{d:>9.2f}{distance_ly(p):>9.1f}"
              f"{mu:>11.2f}{v:>9.1f}k")

    print("\n  Parallax is pure geometry -- the star's apparent shift over Earth's orbit")
    print("  -- and the first rung of the distance ladder Gaia has now climbed for over")
    print("  a billion stars. Proper motion adds the sideways drift: Barnard's Star,")
    print("  the fastest, crosses a Moon's width of sky every ~180 years and moves at")
    print("  142 km/s through space once its radial and tangential motions combine.")

    _svg(os.path.join(outdir, "parallax.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'parallax.svg')}")


def _svg(path, size=720, pad=60):
    cx, cy = size * 0.28, size / 2.0
    r_orbit = 70.0        # Earth's orbit radius on screen
    star_x = size - pad - 30
    star_y = cy

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # Earth's orbit around the Sun
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_orbit:.1f}" '
                 f'fill="none" stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="#ffd43b"/>')
    parts.append(f'<text x="{cx:.1f}" y="{cy+22:.1f}" fill="#ffd43b" '
                 f'font-size="11" text-anchor="middle">Sun</text>')
    # Earth at two opposite points (6 months apart)
    for ey, lbl in [(cy - r_orbit, "Jan"), (cy + r_orbit, "Jul")]:
        parts.append(f'<circle cx="{cx:.1f}" cy="{ey:.1f}" r="4" fill="#4dabf7"/>')
        # sight line to the star
        parts.append(f'<line x1="{cx:.1f}" y1="{ey:.1f}" x2="{star_x:.1f}" '
                     f'y2="{star_y:.1f}" stroke="#4dabf7" stroke-width="0.8" '
                     f'stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{cx-10:.1f}" y="{cy-r_orbit-8:.1f}" fill="#4dabf7" '
                 f'font-size="10" text-anchor="end">Earth (Jan)</text>')
    parts.append(f'<text x="{cx-10:.1f}" y="{cy+r_orbit+14:.1f}" fill="#4dabf7" '
                 f'font-size="10" text-anchor="end">Earth (Jul)</text>')
    # the star
    parts.append(f'<circle cx="{star_x:.1f}" cy="{star_y:.1f}" r="5" fill="#ff6b6b"/>')
    parts.append(f'<text x="{star_x:.1f}" y="{star_y-12:.1f}" fill="#ff6b6b" '
                 f'font-size="11" text-anchor="middle">nearby star</text>')
    # parallax angle arc marker at the star
    parts.append(f'<text x="{star_x-70:.1f}" y="{star_y+30:.1f}" fill="#8b949e" '
                 f'font-size="11">parallax angle p</text>')
    # baseline label
    parts.append(f'<text x="{cx+14:.1f}" y="{cy:.1f}" fill="#8b949e" '
                 f'font-size="10">1 AU baseline</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Stellar parallax: distance from Earth-orbit geometry</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'the star shifts by p over six months; d (pc) = 1 / p (arcsec)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
