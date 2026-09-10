"""Demo: escape and cosmic velocities across the Solar System and beyond.

Tabulates orbital and escape speeds for planets, the Sun, and a white dwarf,
shows the sqrt(2) ratio and the Schwarzschild-radius connection, and renders
escape speed vs surface gravity to SVG.

    python examples/cosmic_velocities_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cosmic_velocities import (orbital_speed, escape_speed,  # noqa: E402
                               solar_system_escape_from_earth_orbit,
                               schwarzschild_radius_from_escape, C)

# (name, mass kg, radius m)
BODIES = [
    ("Moon", 7.342e22, 1.737e6),
    ("Mars", 6.417e23, 3.390e6),
    ("Earth", 5.972e24, 6.371e6),
    ("Jupiter", 1.898e27, 6.991e7),
    ("Sun", 1.98892e30, 6.957e8),
    ("white dwarf", 1.2 * 1.98892e30, 6.371e6),
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Escape and cosmic velocities\n")
    print(f"  {'body':<14}{'v_orbit (km/s)':>16}{'v_escape (km/s)':>17}")
    print("  " + "-" * 47)
    for name, M, R in BODIES:
        print(f"  {name:<14}{orbital_speed(M, R)/1e3:>16.2f}{escape_speed(M, R)/1e3:>17.2f}")
    print(f"\n  escape / orbital = sqrt(2) always.")
    print(f"  leaving the Solar System from Earth's orbit: "
          f"{solar_system_escape_from_earth_orbit()/1e3:.1f} km/s")
    print(f"  set v_escape = c and you get the Schwarzschild radius: "
          f"{schwarzschild_radius_from_escape(1.98892e30):.0f} m for the Sun.")

    _svg(outdir + "/cosmic_velocities.svg")
    print(f"\n  wrote {outdir}/cosmic_velocities.svg")


def _svg(path, size=720, pad=64):
    names = [b[0] for b in BODIES]
    vesc = [escape_speed(b[1], b[2]) / 1e3 for b in BODIES]
    vmax = max(vesc) * 1.15
    n = len(BODIES)
    bw = (size - 2 * pad) / n * 0.6
    gap = (size - 2 * pad) / n

    def bx(i):
        return pad + i * gap + gap * 0.2

    def by(v):
        return size - pad - v / vmax * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    for i, (name, v) in enumerate(zip(names, vesc)):
        parts.append(f'<rect x="{bx(i):.1f}" y="{by(v):.1f}" width="{bw:.1f}" '
                     f'height="{size-pad-by(v):.1f}" fill="#4cc9f0"/>')
        parts.append(f'<text x="{bx(i)+bw/2:.1f}" y="{size-pad+16:.1f}" fill="#8b949e" '
                     f'font-size="10" text-anchor="middle">{name}</text>')
        parts.append(f'<text x="{bx(i)+bw/2:.1f}" y="{by(v)-6:.1f}" fill="#e6edf3" '
                     f'font-size="10" text-anchor="middle">{v:.0f}</text>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Escape velocity by body (km/s)</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'v_escape = sqrt(2) v_orbital; a 1.2 M_sun white dwarf reaches thousands of km/s</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
