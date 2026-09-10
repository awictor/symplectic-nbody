"""Demo: the Knudsen number -- when a gas stops behaving like a fluid.

Prints the mean free path and flow regime across systems (airliner to MEMS channel to
satellite), then draws the regime map over system size and pressure: the continuum, slip,
transitional and free-molecular bands, with real systems and a few altitudes marked.

    python examples/knudsen_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knudsen import (mean_free_path, knudsen_number, flow_regime,  # noqa: E402
                     mean_speed)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Knudsen number Kn = lambda / L: continuum fluid vs free-flying molecules\n")
    print("  Air mean free path: %.1f nm at sea level, mean speed %.0f m/s (300 K)\n"
          % (mean_free_path(300.0, 101325.0) * 1e9, mean_speed(300.0)))
    print(f"  {'system':<26}{'L':>9}{'P':>11}{'Kn':>11}{'regime':>16}")
    # (name, L m, P Pa)
    systems = [
        ("airliner wing", 3.0, 101325.0),
        ("insect wing", 1e-3, 101325.0),
        ("MEMS microchannel", 1e-6, 101325.0),
        ("nanopore filter", 5e-9, 101325.0),
        ("satellite (100 km)", 1.0, 0.03),
        ("vacuum chamber", 0.1, 1e-3),
    ]
    for name, L, P in systems:
        Kn = knudsen_number(L, 300.0, P)
        print(f"  {name:<26}{L:>9.1e}{P:>11.1e}{Kn:>11.2e}{flow_regime(Kn):>16}")

    print("\n  Everyday air is a flawless fluid because lambda ~ 68 nm is dwarfed by anything")
    print("  we touch. Shrink to a chip's cooling pores, or thin the air at orbital altitude,")
    print("  and Kn climbs past 1: the gas slips at walls and finally flies molecule-to-molecule.")

    _svg(os.path.join(outdir, "knudsen.svg"), systems)
    print(f"\n  wrote {os.path.join(outdir, 'knudsen.svg')}")


def _svg(path, systems, size=720, pad=82):
    # regime map: x = log system size (1 nm .. 10 m), y = log pressure (1e-4 .. 1e6 Pa)
    lx0, lx1 = math.log10(1e-9), math.log10(10.0)
    ly0, ly1 = math.log10(1e-4), math.log10(1e6)
    nx, ny = 96, 96

    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 44

    def X(lL):
        return x0 + (lL - lx0) / (lx1 - lx0) * (x1 - x0)

    def Y(lP):
        return y0 - (lP - ly0) / (ly1 - ly0) * (y0 - y1)   # high pressure at bottom

    colors = {"continuum": "#4dabf7", "slip": "#06d6a0",
              "transitional": "#ff922b", "free molecular": "#ff6b6b"}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Rarefied-gas regime map</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'Knudsen number over system size and pressure: fluid (blue) to free-molecular (red)</text>',
    ]

    cw = (x1 - x0) / nx
    ch = (y0 - y1) / ny
    for j in range(ny):
        lP = ly0 + (ly1 - ly0) * (j + 0.5) / ny
        P = 10.0 ** lP
        yy = Y(lP) - ch
        for i in range(nx):
            lL = lx0 + (lx1 - lx0) * (i + 0.5) / nx
            L = 10.0 ** lL
            Kn = knudsen_number(L, 300.0, P)
            parts.append(f'<rect x="{X(lL)-cw/2:.1f}" y="{yy:.1f}" width="{cw+0.8:.1f}" '
                         f'height="{ch+0.8:.1f}" fill="{colors[flow_regime(Kn)]}" opacity="0.7"/>')

    # frame + ticks
    parts.append(f'<rect x="{x0:.1f}" y="{y1:.1f}" width="{x1-x0:.1f}" height="{y0-y1:.1f}" '
                 f'fill="none" stroke="#8b949e" stroke-width="1.5"/>')
    for e in range(-9, 2, 2):
        gx = X(e)
        parts.append(f'<text x="{gx:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">10^{e} m</text>')
    for e in range(-4, 7, 2):
        gy = Y(e)
        parts.append(f'<text x="{x0-6:.1f}" y="{gy+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">10^{e}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+30:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">system size L (m)</text>')
    parts.append(f'<text x="24" y="{(y0+y1)/2:.1f}" fill="#8b949e" font-size="11" '
                 f'transform="rotate(-90 24 {(y0+y1)/2:.1f})" text-anchor="middle">pressure (Pa)</text>')
    # sea-level line
    parts.append(f'<line x1="{x0:.1f}" y1="{Y(math.log10(101325.0)):.1f}" x2="{x1:.1f}" '
                 f'y2="{Y(math.log10(101325.0)):.1f}" stroke="#8b949e" stroke-width="0.8" '
                 f'stroke-dasharray="3 4" opacity="0.6"/>')
    parts.append(f'<text x="{x1-4:.1f}" y="{Y(math.log10(101325.0))-4:.1f}" fill="#8b949e" '
                 f'font-size="9" text-anchor="end">sea level</text>')

    # plot systems
    for name, L, P in systems:
        gx, gy = X(math.log10(L)), Y(math.log10(P))
        parts.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="4.5" fill="#e6edf3" '
                     f'stroke="#0d1117" stroke-width="1.5"/>')
        parts.append(f'<text x="{gx+8:.1f}" y="{gy+4:.1f}" fill="#e6edf3" font-size="10">{name}</text>')

    # legend
    ly = y1 - 2
    for i, (reg, col) in enumerate([("continuum", colors["continuum"]),
                                    ("slip", colors["slip"]),
                                    ("transitional", colors["transitional"]),
                                    ("free molecular", colors["free molecular"])]):
        lxp = x0 + i * 150
        parts.append(f'<rect x="{lxp:.1f}" y="{ly-10:.1f}" width="12" height="12" fill="{col}"/>')
        parts.append(f'<text x="{lxp+16:.1f}" y="{ly:.1f}" fill="#8b949e" font-size="10">{reg}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
