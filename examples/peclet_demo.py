"""Demo: the Peclet number -- advection vs diffusion across systems.

Prints the Peclet number of everyday transport (cell, capillary, stirred cup, river) plus
the Prandtl/Schmidt/Lewis numbers of water and air, then draws the advection-diffusion regime
map: flow speed (x) vs length scale (y) on log-log axes, shaded by Peclet number with the
Pe = 1 crossover line and several real systems marked.

    python examples/peclet_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from peclet import (peclet, thermal_diffusivity, prandtl, schmidt, lewis,  # noqa: E402
                    crossover_length)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    D = 1e-9   # small-molecule mass diffusivity in water
    print("Peclet number Pe = U L / D: advection (carried) vs diffusion (spreads)\n")
    print(f"  {'system':<30}{'U (m/s)':>10}{'L (m)':>10}{'Pe':>14}{'regime':>14}")
    systems = [
        ("inside a cell", 1e-7, 1e-5),
        ("blood in a capillary", 5e-4, 8e-6),
        ("slowly stirred cup", 0.05, 0.05),
        ("stream / small river", 0.5, 1.0),
    ]
    for label, U, L in systems:
        p = peclet(U, L, D)
        regime = "diffusion" if p < 1 else "advection"
        print(f"  {label:<30}{U:>10.1e}{L:>10.1e}{p:>14.2e}{regime:>14}")

    alpha_w = thermal_diffusivity(0.6, 1000.0, 4180.0)
    alpha_a = thermal_diffusivity(0.026, 1.2, 1005.0)
    print("\n  Fluid diffusivity ratios:")
    print(f"    water:  Pr = nu/alpha = {prandtl(1e-6, alpha_w):.1f}   "
          f"Sc = nu/D = {schmidt(1e-6, D):.0f}   Le = alpha/D = {lewis(alpha_w, D):.0f}")
    print(f"    air:    Pr = {prandtl(1.5e-5, alpha_a):.2f}   (heat and momentum spread alike)")

    print("\n  Pe = Re*Pr for heat and Re*Sc for mass. The crossover Pe=1 sits at L = D/U:")
    for U in (1e-6, 1e-3, 1.0):
        print(f"    U = {U:>8.0e} m/s  ->  crossover length = {crossover_length(U, D):.2e} m")
    print("  Below that length diffusion wins; above it the flow carries faster than spreading.")

    _svg(os.path.join(outdir, "peclet.svg"), D, systems)
    print(f"\n  wrote {os.path.join(outdir, 'peclet.svg')}")


def _svg(path, D, systems, size=720, pad=82):
    # regime map: x = log10 velocity (1e-8 .. 1e1 m/s), y = log10 length (1e-7 .. 1e2 m)
    lu0, lu1 = -8.0, 1.0
    ll0, ll1 = -7.0, 2.0
    nx, ny = 84, 84

    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 40

    def X(lu):
        return x0 + (lu - lu0) / (lu1 - lu0) * (x1 - x0)

    def Y(ll):
        return y0 - (ll - ll0) / (ll1 - ll0) * (y0 - y1)   # length increases upward

    def color(logPe):
        # logPe from about -6..+9; blue (diffusion) -> gray (~1) -> orange (advection)
        t = max(-4.0, min(8.0, logPe))
        if t < 0:
            u = (t + 4.0) / 4.0            # -4..0 -> 0..1 (blue -> gray)
            r = int(0x1b + u * (0x8b - 0x1b)); g = int(0x3a + u * (0x94 - 0x3a)); b = int(0x8b + u * (0x9d - 0x8b))
        else:
            u = t / 8.0                    # 0..8 -> gray -> orange
            r = int(0x8b + u * (0xff - 0x8b)); g = int(0x94 + u * (0x92 - 0x94)); b = int(0x9d + u * (0x2b - 0x9d))
        return f"#{r:02x}{g:02x}{b:02x}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Advection-diffusion regime map</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'Peclet = U L / D (aqueous solute); blue = diffusion wins, orange = advection wins</text>',
    ]

    cw = (x1 - x0) / nx
    ch = (y0 - y1) / ny
    for j in range(ny):
        ll = ll0 + (ll1 - ll0) * (j + 0.5) / ny
        yy = Y(ll) - ch
        for i in range(nx):
            lu = lu0 + (lu1 - lu0) * (i + 0.5) / nx
            logPe = lu + ll - math.log10(D)
            parts.append(f'<rect x="{X(lu)-cw/2:.1f}" y="{yy:.1f}" width="{cw+0.8:.1f}" '
                         f'height="{ch+0.8:.1f}" fill="{color(logPe)}"/>')

    # Pe = 1 crossover line: lu + ll = log10(D)  ->  ll = log10(D) - lu
    logD = math.log10(D)
    xa, ya = X(lu0), Y(logD - lu0)
    xb, yb = X(lu1), Y(logD - lu1)
    parts.append(f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" '
                 f'stroke="#ffd43b" stroke-width="2.4" stroke-dasharray="7 5"/>')
    parts.append(f'<text x="{X(-2.5):.1f}" y="{Y(logD+2.5)-6:.1f}" fill="#ffd43b" '
                 f'font-size="12" transform="rotate(-33 {X(-2.5):.1f} {Y(logD+2.5):.1f})">Pe = 1 crossover</text>')

    # frame + axes ticks
    parts.append(f'<rect x="{x0:.1f}" y="{y1:.1f}" width="{x1-x0:.1f}" height="{y0-y1:.1f}" '
                 f'fill="none" stroke="#8b949e" stroke-width="1.5"/>')
    for e in range(-8, 2):
        gx = X(e)
        parts.append(f'<text x="{gx:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">10^{e}</text>')
    for e in range(-7, 3):
        gy = Y(e)
        parts.append(f'<text x="{x0-6:.1f}" y="{gy+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">10^{e}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+30:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">flow speed U (m/s)</text>')
    parts.append(f'<text x="20" y="{(y0+y1)/2:.1f}" fill="#8b949e" font-size="11" '
                 f'transform="rotate(-90 20 {(y0+y1)/2:.1f})" text-anchor="middle">length scale L (m)</text>')

    # plot the real systems
    for label, U, L in systems:
        gx, gy = X(math.log10(U)), Y(math.log10(L))
        parts.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="5" fill="#e6edf3" stroke="#0d1117" stroke-width="1.5"/>')
        parts.append(f'<text x="{gx+9:.1f}" y="{gy+4:.1f}" fill="#e6edf3" font-size="11">{label}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
