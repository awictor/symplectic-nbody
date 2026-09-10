"""Demo: the Ekman spiral -- wind-driven current turning with depth.

Prints the current speed and direction at increasing depth, then draws the Ekman spiral as
a hodograph: the tip of the velocity vector traces a shrinking clockwise spiral, starting
45 degrees to the right of the wind at the surface (northern hemisphere).

    python examples/ekman_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ekman import (coriolis_parameter, ekman_depth, velocity_at_depth,  # noqa: E402
                   ekman_transport, RHO_SEAWATER)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    lat, A_z, tau = 45.0, 0.05, 0.1        # 45 N, eddy visc 0.05 m^2/s, wind stress 0.1 Pa
    f = coriolis_parameter(lat)
    D = ekman_depth(A_z, f)

    print("Ekman spiral: wind-driven ocean boundary layer at %g N\n" % lat)
    print("  eddy viscosity A_z = %.2f m^2/s   wind stress tau = %.2f Pa" % (A_z, tau))
    print("  Coriolis f = %.3e /s   Ekman depth D = %.1f m\n" % (f, D))
    print("  Current spirals CLOCKWISE, decaying with depth (wind blows toward +x/east):")
    print(f"  {'depth':>8}{'speed':>12}{'direction':>26}")
    for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        z = -frac * D
        u, v = velocity_at_depth(z, tau, A_z, f)
        spd = math.hypot(u, v)
        ang = math.degrees(math.atan2(v, u))
        rel = ang  # wind is along +x, so angle-from-wind = ang
        print(f"  {z:>6.1f} m{spd*100:>9.2f} cm/s{rel:>18.0f} deg from wind")

    T = ekman_transport(tau, f)
    print("\n  Net Ekman transport = %.3f m^2/s, directed 90 deg to the RIGHT of the wind."
          % T)
    print("  Surface current sits 45 deg right of the wind; each deeper layer is dragged")
    print("  further right and weaker, so the vertical sum points fully across the wind --")
    print("  the sideways pumping that drives coastal upwelling and the ocean gyres.")

    _svg(os.path.join(outdir, "ekman.svg"), tau, A_z, f)
    print(f"\n  wrote {os.path.join(outdir, 'ekman.svg')}")


def _svg(path, tau, A_z, f, size=720):
    # Hodograph: plot (u, v) for z from 0 down to -1.5 D as a spiral, tails at origin.
    D = ekman_depth(A_z, f)
    n = 240
    pts = []
    for i in range(n + 1):
        z = -1.5 * D * i / n
        u, v = velocity_at_depth(z, tau, A_z, f)
        pts.append((u, v))

    vmax = max(math.hypot(u, v) for u, v in pts) * 1.15
    cx, cy = size * 0.52, size * 0.55
    scale = (size * 0.36) / vmax

    def X(u):
        return cx + u * scale

    def Y(v):
        return cy - v * scale     # screen y is down; +v (north) points up

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The Ekman spiral (hodograph)</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'current vector turns clockwise and shrinks with depth; 45 deg right of wind at surface</text>',
    ]

    # axes through origin
    parts.append(f'<line x1="{cx-size*0.42:.1f}" y1="{cy:.1f}" x2="{cx+size*0.42:.1f}" y2="{cy:.1f}" '
                 f'stroke="#21262d" stroke-width="1"/>')
    parts.append(f'<line x1="{cx:.1f}" y1="{cy-size*0.42:.1f}" x2="{cx:.1f}" y2="{cy+size*0.42:.1f}" '
                 f'stroke="#21262d" stroke-width="1"/>')

    # wind arrow (along +x / east)
    wx = cx + size * 0.40
    parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{wx:.1f}" y2="{cy:.1f}" '
                 f'stroke="#ffd43b" stroke-width="2"/>')
    parts.append(f'<polygon points="{wx:.1f},{cy:.1f} {wx-12:.1f},{cy-6:.1f} {wx-12:.1f},{cy+6:.1f}" '
                 f'fill="#ffd43b"/>')
    parts.append(f'<text x="{wx-4:.1f}" y="{cy-10:.1f}" fill="#ffd43b" font-size="12" '
                 f'text-anchor="end">wind</text>')

    # the spiral (tip of velocity vector over depth)
    poly = " ".join(f"{X(u):.1f},{Y(v):.1f}" for u, v in pts)
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4dabf7" stroke-width="2.4"/>')

    # velocity vectors at a few depths, tails at origin
    for frac, col in ((0.0, "#06d6a0"), (0.25, "#8338ec"), (0.5, "#ff922b"), (1.0, "#ff6b6b")):
        z = -frac * D
        u, v = velocity_at_depth(z, tau, A_z, f)
        parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{X(u):.1f}" y2="{Y(v):.1f}" '
                     f'stroke="{col}" stroke-width="1.8" opacity="0.9"/>')
        parts.append(f'<circle cx="{X(u):.1f}" cy="{Y(v):.1f}" r="4" fill="{col}"/>')
        lbl = "surface" if frac == 0.0 else f"z=-{frac:g}D"
        parts.append(f'<text x="{X(u)+7:.1f}" y="{Y(v)-6:.1f}" fill="{col}" font-size="11">{lbl}</text>')

    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="#8b949e"/>')
    parts.append(f'<text x="{cx-size*0.40:.1f}" y="{cy+size*0.40:.1f}" fill="#8b949e" '
                 f'font-size="11">axes: along-wind (x) & cross-wind (y) current components</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
