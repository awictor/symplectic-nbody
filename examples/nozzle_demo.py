"""Demo: the de Laval nozzle -- squeezing gas subsonic, expanding it supersonic.

Prints the isentropic ratios and exit Mach numbers for a range of area ratios, then draws
the nozzle: a converging-diverging bell with the Mach number climbing through 1 at the throat
and the pressure falling monotonically, the signature of choked supersonic flow.

    python examples/nozzle_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nozzle import (area_ratio, mach_from_area_ratio, pressure_ratio,  # noqa: E402
                    critical_pressure_ratio, choked_mass_flow, exhaust_velocity)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("de Laval nozzle: converge to sonic throat, diverge to supersonic exhaust\n")
    print("  Air (gamma=1.4). Choking pressure ratio P*/P0 = %.4f\n" % critical_pressure_ratio())
    print(f"  {'area ratio A_e/A*':>18}{'exit Mach':>12}{'P_e/P0':>12}")
    for ar in (1.0, 2.0, 4.0, 10.0, 25.0, 100.0):
        Me = mach_from_area_ratio(ar, supersonic=True)
        print(f"  {ar:>18.1f}{Me:>12.2f}{1.0/pressure_ratio(Me):>12.4f}")

    print("\n  Rocket-ish chamber: P0 = 5 MPa, T0 = 3000 K, throat A* = 10 cm^2:")
    mdot = choked_mass_flow(5e6, 3000.0, 10e-4)
    Me = mach_from_area_ratio(25.0, supersonic=True)
    ve = exhaust_velocity(3000.0, Me)
    print(f"    choked mass flow  = {mdot:.2f} kg/s")
    print(f"    area ratio 25 -> exit Mach {Me:.2f}, exhaust {ve:.0f} m/s")

    print("\n  Subsonic flow accelerates as area shrinks, but past Mach 1 it reverses: a")
    print("  supersonic stream speeds up as the area GROWS, so the throat must sit exactly at")
    print("  Mach 1. Once choked, mass flow is capped by the throat and only the bell sets M_e.")

    _svg(os.path.join(outdir, "nozzle.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'nozzle.svg')}")


def _svg(path, size=720, pad=70):
    # Build a nozzle: x from inlet (subsonic, big A) through throat (A=1) to exit (supersonic).
    # Parameterize by Mach from 0.15 -> 3.0; convert to area ratio for the wall shape.
    n = 160
    machs = [0.15 + (3.0 - 0.15) * i / (n - 1) for i in range(n)]
    ars = [area_ratio(m) for m in machs]
    # place throat (min area) partway along; x proportional to index
    armax = max(ars)

    x0, x1 = pad, size - pad
    axis = size * 0.40
    hscale = (size * 0.24) / armax    # half-height per unit area ratio

    def X(i):
        return x0 + (x1 - x0) * i / (n - 1)

    def wall_half(ar):
        return ar * hscale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The de Laval nozzle</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'flow chokes at the throat (M=1); Mach climbs and pressure falls into the diverging bell</text>',
    ]

    # nozzle walls (mirror about axis)
    up = " ".join(f"{X(i):.1f},{axis - wall_half(ars[i]):.1f}" for i in range(n))
    dn = " ".join(f"{X(i):.1f},{axis + wall_half(ars[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{up}" fill="none" stroke="#8b949e" stroke-width="2.4"/>')
    parts.append(f'<polyline points="{dn}" fill="none" stroke="#8b949e" stroke-width="2.4"/>')

    # throat marker (min area index)
    ti = min(range(n), key=lambda i: ars[i])
    parts.append(f'<line x1="{X(ti):.1f}" y1="{axis - wall_half(ars[ti]):.1f}" '
                 f'x2="{X(ti):.1f}" y2="{axis + wall_half(ars[ti]):.1f}" '
                 f'stroke="#ffd43b" stroke-width="1.5" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{X(ti):.1f}" y="{axis - wall_half(armax) - 6:.1f}" fill="#ffd43b" '
                 f'font-size="11" text-anchor="middle">throat  M = 1</text>')
    parts.append(f'<text x="{x0+4:.1f}" y="{axis+4:.1f}" fill="#8b949e" font-size="10">subsonic</text>')
    parts.append(f'<text x="{x1-4:.1f}" y="{axis+4:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end">supersonic</text>')

    # Mach and pressure curves below the nozzle
    py0, py1 = size * 0.66, size - pad
    mmax = max(machs)
    def MY(m):
        return py1 - (m / mmax) * (py1 - py0)
    def PY(p):
        return py1 - p * (py1 - py0)      # p is P/P0 in 0..1
    mpoly = " ".join(f"{X(i):.1f},{MY(machs[i]):.1f}" for i in range(n))
    ppoly = " ".join(f"{X(i):.1f},{PY(1.0/pressure_ratio(machs[i])):.1f}" for i in range(n))
    parts.append(f'<polyline points="{mpoly}" fill="none" stroke="#4dabf7" stroke-width="2.4"/>')
    parts.append(f'<polyline points="{ppoly}" fill="none" stroke="#ff6b6b" stroke-width="2.4"/>')
    # M=1 line on the mach plot
    parts.append(f'<line x1="{x0:.1f}" y1="{MY(1.0):.1f}" x2="{x1:.1f}" y2="{MY(1.0):.1f}" '
                 f'stroke="#4dabf7" stroke-width="0.8" stroke-dasharray="3 4" opacity="0.5"/>')
    parts.append(f'<line x1="{X(ti):.1f}" y1="{py0:.1f}" x2="{X(ti):.1f}" y2="{py1:.1f}" '
                 f'stroke="#ffd43b" stroke-width="0.8" stroke-dasharray="4 3" opacity="0.5"/>')
    parts.append(f'<text x="{x0+4:.1f}" y="{MY(1.0)-4:.1f}" fill="#4dabf7" font-size="10">M = 1</text>')
    parts.append(f'<text x="{x0+4:.1f}" y="{py0+12:.1f}" fill="#4dabf7" font-size="11">Mach number</text>')
    parts.append(f'<text x="{x1-4:.1f}" y="{py0+12:.1f}" fill="#ff6b6b" font-size="11" '
                 f'text-anchor="end">pressure P/P0</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
