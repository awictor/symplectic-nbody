"""Demo: orbits around a Schwarzschild black hole.

Locates the ISCO (6M) and photon sphere (3M), then renders a strongly precessing
bound orbit and a plunging orbit around the hole, with the horizon, photon
sphere, and ISCO drawn as reference circles.

    python examples/schwarzschild_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schwarzschild import (isco_radius, photon_sphere_radius, orbit_shape,  # noqa: E402
                           precession_per_orbit)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    M = 1.0
    print("Orbits around a Schwarzschild black hole (units of M)\n")
    print(f"  event horizon  : r = {2*M:.0f} M")
    print(f"  photon sphere  : r = {photon_sphere_radius(M):.0f} M")
    print(f"  ISCO           : r = {isco_radius(M):.0f} M\n")

    # strongly precessing bound orbit (close in, so precession is large)
    prec = precession_per_orbit(10.0, 20.0, M)
    print(f"  bound orbit (r: 10-20 M): perihelion advance {math.degrees(prec):.1f} deg/orbit")
    print(f"    (Mercury's is 43 arcsec/CENTURY; here it's tens of degrees PER ORBIT)\n")

    bound_phis, bound_rs = orbit_shape(20.0, 4.3, 0.0, M, dphi=5e-4, max_phi=60.0)
    plunge_phis, plunge_rs = orbit_shape(12.0, 3.4, 0.0, M, dphi=5e-4, max_phi=40.0)
    print(f"  plunging orbit: falls from r=12 M through the horizon to r={min(plunge_rs):.2f} M")

    _svg(bound_phis, bound_rs, plunge_phis, plunge_rs, M,
         os.path.join(outdir, "schwarzschild.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'schwarzschild.svg')}")
    print("  The bound orbit is a precessing rosette; the plunger spirals through")
    print("  the horizon. Both are exact Schwarzschild geodesics.")


def _svg(bp, br, pp, pr, M, path, size=720):
    # polar (r, phi) -> cartesian
    def pts(phis, rs):
        return [(rs[i] * math.cos(phis[i]), rs[i] * math.sin(phis[i]))
                for i in range(len(rs))]

    bpts = pts(bp, br)
    ppts = pts(pp, pr)
    ext = 21.0

    def sx(x): return size / 2 + x / ext * (size / 2 - 20)
    def sy(y): return size / 2 - y / ext * (size / 2 - 20)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#05070d"/>',
    ]
    # reference circles
    for radius, col, dash in ((2 * M, "#ffffff", "none"),
                              (photon_sphere_radius(M), "#f4a261", "3,3"),
                              (isco_radius(M), "#4cc9f0", "5,4")):
        parts.append(f'<circle cx="{sx(0):.1f}" cy="{sy(0):.1f}" '
                     f'r="{radius/ext*(size/2-20):.1f}" fill="none" stroke="{col}" '
                     f'stroke-dasharray="{dash}" stroke-width="1"/>')
    # black hole
    parts.append(f'<circle cx="{sx(0):.1f}" cy="{sy(0):.1f}" '
                 f'r="{2*M/ext*(size/2-20):.1f}" fill="#000000"/>')
    # orbits
    poly_b = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in bpts)
    poly_p = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in ppts)
    parts.append(f'<polyline points="{poly_b}" fill="none" stroke="#2a9d8f" stroke-width="1.2"/>')
    parts.append(f'<polyline points="{poly_p}" fill="none" stroke="#ff006e" stroke-width="1.2"/>')
    parts.append(f'<text x="16" y="26" fill="#e6edf3" font-size="16">'
                 f'Schwarzschild orbits: precessing (teal) &amp; plunging (pink)</text>')
    parts.append(f'<text x="16" y="{size-16}" fill="#8b949e" font-size="11">'
                 f'white=horizon(2M), orange=photon sphere(3M), blue=ISCO(6M)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
