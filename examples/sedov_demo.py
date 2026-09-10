"""Demo: the Sedov-Taylor blast wave (supernova remnants and the Trinity test).

Traces a supernova remnant's radius and shock speed over time (R ~ t^2/5), and
reproduces G. I. Taylor's trick of recovering the Trinity bomb's yield from the
fireball's radius-vs-time. Renders the expansion history to SVG.

    python examples/sedov_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sedov import (shock_radius, shock_velocity, energy_from_radius,  # noqa: E402
                   shock_temperature)

PC = 3.086e16
YR = 3.156e7
KT = 4.184e12


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    E, rho = 1e44, 2e-21  # 1e51 erg into ~1 proton/cc ISM

    print("Sedov-Taylor blast wave\n")
    print("  Trinity test (Taylor's declassification trick):")
    kt = energy_from_radius(130.0, 1.25, 0.025) / KT
    print(f"    fireball R=130 m at t=25 ms in air -> yield ~ {kt:.0f} kilotons")
    print(f"    (the actual device was ~21 kt -- right order from a movie frame)\n")

    print("  Supernova remnant (E=1e51 erg, ISM ~1 H/cc):")
    print(f"  {'age (yr)':>10}{'radius (pc)':>14}{'shock (km/s)':>14}{'T (K)':>12}")
    print("  " + "-" * 50)
    ages_yr = [100, 300, 1000, 3000, 10000]
    curve = []
    for age in ages_yr:
        t = age * YR
        R = shock_radius(E, rho, t)
        v = shock_velocity(E, rho, t)
        T = shock_temperature(E, rho, t)
        curve.append((age, R / PC, v / 1e3))
        print(f"  {age:>10}{R/PC:>14.2f}{v/1e3:>14.0f}{T:>12.1e}")
    print("\n  R grows as t^2/5 and the shock decelerates as t^-3/5. The same self-")
    print("  similar law dates supernova remnants and (run backwards) weighed the bomb.")

    _svg(curve, os.path.join(outdir, "sedov.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'sedov.svg')}")


def _svg(curve, path, size=720, pad=64):
    amax = max(a for a, _, _ in curve)
    rmax = max(r for _, r, _ in curve) * 1.1
    vmax = max(v for _, _, v in curve) * 1.1

    def sx(a):
        return pad + a / amax * (size - 2 * pad)

    def sy_r(r):
        return size - pad - r / rmax * (size - 2 * pad)

    def sy_v(v):
        return size - pad - v / vmax * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    rpoly = " ".join(f"{sx(a):.1f},{sy_r(r):.1f}" for a, r, _ in curve)
    vpoly = " ".join(f"{sx(a):.1f},{sy_v(v):.1f}" for a, _, v in curve)
    parts.append(f'<polyline points="{rpoly}" fill="none" stroke="#4cc9f0" stroke-width="2"/>')
    parts.append(f'<polyline points="{vpoly}" fill="none" stroke="#ff006e" stroke-width="2"/>')
    for a, r, v in curve:
        parts.append(f'<circle cx="{sx(a):.1f}" cy="{sy_r(r):.1f}" r="3" fill="#4cc9f0"/>')
        parts.append(f'<circle cx="{sx(a):.1f}" cy="{sy_v(v):.1f}" r="3" fill="#ff006e"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Supernova remnant: radius (blue) &amp; shock speed (pink)</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+8}" fill="#4cc9f0" font-size="12">'
                 f'radius ~ t^2/5 (rises)</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+26}" fill="#ff006e" font-size="12">'
                 f'shock speed ~ t^-3/5 (decelerates)</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">remnant age (yr) -&gt;</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
