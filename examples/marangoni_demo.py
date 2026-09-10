"""Demo: the Marangoni effect -- flow driven by a surface-tension gradient.

Prints the Marangoni number and convection onset for heated liquid layers and the surface
stress of a tears-of-wine gradient, then draws the Marangoni-vs-buoyancy regime map over
layer thickness and gravity: thin layers and microgravity are surface-tension-driven, thick
layers on the ground are buoyancy-driven.

    python examples/marangoni_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from marangoni import (marangoni_thermal, is_convecting, surface_stress,  # noqa: E402
                       dynamic_bond_number, marangoni_velocity, MA_CRITICAL)


DG_DT = -1.5e-4     # water surface tension vs temperature (N/m/K)
MU, ALPHA = 1e-3, 1.4e-7
RHO, BETA = 998.0, 2.1e-4


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Marangoni effect: flow from a surface-tension gradient (Ma_c ~ %g)\n" % MA_CRITICAL)
    print("  Heated water layer, dT = 10 K (dgamma/dT = -1.5e-4 N/m/K):\n")
    print(f"  {'layer L':>10}{'Ma':>12}{'onset?':>10}{'flow U':>12}")
    for L in (5e-5, 2e-4, 1e-3, 5e-3):
        Ma = marangoni_thermal(DG_DT, 10.0, L, MU, ALPHA)
        U = marangoni_velocity(DG_DT, 10.0, L, MU)
        yn = "convects" if is_convecting(Ma) else "quiescent"
        print(f"  {L*1000:>7.2f} mm{Ma:>12.1e}{yn:>10}{U:>10.2f} m/s")

    print("\n  Tears of wine: ethanol evaporates from the film climbing the glass, raising its")
    print("  surface tension there, so liquid is pulled UP toward the higher-tension rim until")
    print("  it beads and runs back as 'legs'. A pure solutal Marangoni flow, no heat needed.")

    print("\n  Marangoni vs buoyancy (dynamic Bond number Bo_d = Ra/Ma):")
    for L, g, place in ((1e-4, 9.81, "thin film, Earth"),
                        (1e-2, 9.81, "thick pool, Earth"),
                        (1e-2, 1e-4, "thick pool, orbit")):
        Bo = dynamic_bond_number(RHO, g, BETA, L, DG_DT)
        who = "buoyancy" if Bo > 1 else "Marangoni"
        print(f"    {place:<20} Bo_d = {Bo:>9.2e}  ->  {who}-driven")

    _svg(os.path.join(outdir, "marangoni.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'marangoni.svg')}")


def _svg(path, size=720, pad=80):
    # regime map: x = log layer thickness (10 um .. 10 cm), y = log gravity (1e-5 .. 20 g0)
    lx0, lx1 = math.log10(1e-5), math.log10(0.1)
    ly0, ly1 = math.log10(1e-5 * 9.81), math.log10(20.0 * 9.81)
    nx, ny = 90, 90

    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 44

    def X(lL):
        return x0 + (lL - lx0) / (lx1 - lx0) * (x1 - x0)

    def Y(lg):
        return y0 - (lg - ly0) / (ly1 - ly0) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Marangoni vs buoyancy regime map</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'dynamic Bond number Bo_d = rho g beta L^2 / |dgamma/dT|; thin/low-g = surface tension wins</text>',
    ]

    cw = (x1 - x0) / nx
    ch = (y0 - y1) / ny
    for j in range(ny):
        lg = ly0 + (ly1 - ly0) * (j + 0.5) / ny
        g = 10.0 ** lg
        yy = Y(lg) - ch
        for i in range(nx):
            lL = lx0 + (lx1 - lx0) * (i + 0.5) / nx
            L = 10.0 ** lL
            Bo = dynamic_bond_number(RHO, g, BETA, L, DG_DT)
            col = "#ff922b" if Bo > 1.0 else "#4dabf7"     # orange buoyancy, blue Marangoni
            parts.append(f'<rect x="{X(lL)-cw/2:.1f}" y="{yy:.1f}" width="{cw+0.8:.1f}" '
                         f'height="{ch+0.8:.1f}" fill="{col}" opacity="0.75"/>')

    # Bo_d = 1 crossover line: rho g beta L^2 = |dgamma/dT| -> g = |dgamma/dT|/(rho beta L^2)
    cpts = []
    for i in range(nx + 1):
        lL = lx0 + (lx1 - lx0) * i / nx
        L = 10.0 ** lL
        g_cross = abs(DG_DT) / (RHO * BETA * L * L)
        lg = math.log10(g_cross)
        if ly0 <= lg <= ly1:
            cpts.append(f"{X(lL):.1f},{Y(lg):.1f}")
    if cpts:
        parts.append(f'<polyline points="{" ".join(cpts)}" fill="none" stroke="#ffd43b" '
                     f'stroke-width="2.6" stroke-dasharray="7 5"/>')

    # frame + ticks
    parts.append(f'<rect x="{x0:.1f}" y="{y1:.1f}" width="{x1-x0:.1f}" height="{y0-y1:.1f}" '
                 f'fill="none" stroke="#8b949e" stroke-width="1.5"/>')
    for e in range(-5, -1):
        gx = X(e)
        parts.append(f'<text x="{gx:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">10^{e} m</text>')
    # earth-gravity line
    parts.append(f'<line x1="{x0:.1f}" y1="{Y(math.log10(9.81)):.1f}" x2="{x1:.1f}" '
                 f'y2="{Y(math.log10(9.81)):.1f}" stroke="#8b949e" stroke-width="0.8" '
                 f'stroke-dasharray="3 4" opacity="0.6"/>')
    parts.append(f'<text x="{x1-4:.1f}" y="{Y(math.log10(9.81))-4:.1f}" fill="#8b949e" '
                 f'font-size="10" text-anchor="end">Earth gravity</text>')

    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+30:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">layer thickness L</text>')
    parts.append(f'<text x="24" y="{(y0+y1)/2:.1f}" fill="#8b949e" font-size="11" '
                 f'transform="rotate(-90 24 {(y0+y1)/2:.1f})" text-anchor="middle">gravity (m/s^2)</text>')
    parts.append(f'<text x="{X(-4.5):.1f}" y="{Y(math.log10(1e-3)):.1f}" fill="#0d1117" '
                 f'font-size="12" font-weight="bold">Marangoni</text>')
    parts.append(f'<text x="{X(-2.2):.1f}" y="{Y(math.log10(50)):.1f}" fill="#0d1117" '
                 f'font-size="12" font-weight="bold">buoyancy</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
