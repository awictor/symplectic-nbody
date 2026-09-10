"""Demo: surface tension -- capillary rise and Young-Laplace pressure.

Prints how high water climbs tubes of shrinking radius (Jurin's law) and the overpressure
inside droplets and bubbles, then draws capillary rise vs tube radius on log-log axes,
with tubes of water climbing higher as they narrow.

    python examples/surface_tension_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from surface_tension import (capillary_rise, droplet_pressure, bubble_pressure,  # noqa: E402
                             GAMMA_WATER)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Surface tension: gamma = 0.0728 N/m for water at 20 C\n")
    print("  Capillary rise (Jurin's law  h = 2 gamma cos(theta) / (rho g r)):")
    print(f"  {'tube radius':>14}{'rise':>16}")
    for r_mm in (5.0, 1.0, 0.5, 0.1, 0.01, 0.001):
        r = r_mm * 1e-3
        h = capillary_rise(GAMMA_WATER, r)
        if h < 1e-2:
            hs = f"{h*1000:.2f} mm"
        elif h < 1.0:
            hs = f"{h*100:.2f} cm"
        else:
            hs = f"{h:.1f} m"
        print(f"  {r_mm:>10.3f} mm{hs:>16}")

    print("\n  Young-Laplace overpressure  (droplet 2 gamma/r,  bubble 4 gamma/r):")
    print(f"  {'radius':>12}{'droplet':>14}{'soap bubble':>16}")
    for r_mm in (5.0, 1.0, 0.1):
        r = r_mm * 1e-3
        print(f"  {r_mm:>8.1f} mm{droplet_pressure(GAMMA_WATER, r):>10.1f} Pa"
              f"{bubble_pressure(GAMMA_WATER, r):>13.1f} Pa")

    print("\n  Narrower bore -> higher climb (rise ~ 1/r): a 1 micron root pore lifts")
    print("  water tens of metres. Smaller drops hold higher pressure, so when a small")
    print("  bubble meets a big one through a tube, the small one empties into the big.")

    _svg(os.path.join(outdir, "surface_tension.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'surface_tension.svg')}")


def _svg(path, size=720, pad=80):
    # log-log: capillary rise (m) vs tube radius (m) over 1 mm .. 10 nm
    r_min, r_max = 1e-8, 1e-2          # radii in metres (10 nm .. 10 mm)
    n = 160
    lr0, lr1 = math.log10(r_min), math.log10(r_max)
    radii = [10 ** (lr0 + (lr1 - lr0) * i / (n - 1)) for i in range(n)]
    rises = [capillary_rise(GAMMA_WATER, r) for r in radii]

    lx0, lx1 = lr0, lr1
    ly0, ly1 = math.log10(min(rises)), math.log10(max(rises))

    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 40

    def X(r):
        return x0 + (math.log10(r) - lx0) / (lx1 - lx0) * (x1 - x0)

    def Y(h):
        return y0 - (math.log10(h) - ly0) / (ly1 - ly0) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">'
        f'Capillary rise vs tube radius (Jurin&#39;s law)</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'water at 20 C -- narrower bore climbs higher, rise proportional to 1/r</text>',
    ]

    # axes
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.5"/>')

    # x gridlines/labels (decades of radius)
    for e in range(-8, -1):
        r = 10.0 ** e
        gx = X(r)
        parts.append(f'<line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y1}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        lbl = f"10^{e} m"
        parts.append(f'<text x="{gx:.1f}" y="{y0+16:.1f}" fill="#8b949e" '
                     f'font-size="10" text-anchor="middle">{lbl}</text>')
    # y gridlines/labels (decades of rise)
    ey0, ey1 = int(math.floor(ly0)), int(math.ceil(ly1))
    for e in range(ey0, ey1 + 1):
        h = 10.0 ** e
        gy = Y(h)
        parts.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{x0-8:.1f}" y="{gy+4:.1f}" fill="#8b949e" '
                     f'font-size="10" text-anchor="end">10^{e} m</text>')

    # the rise curve
    poly = " ".join(f"{X(radii[i]):.1f},{Y(rises[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')

    # mark the 1 mm-diameter tube (r = 0.5 mm, ~1.5 cm rise)
    r_mark = 0.5e-3
    h_mark = capillary_rise(GAMMA_WATER, r_mark)
    parts.append(f'<circle cx="{X(r_mark):.1f}" cy="{Y(h_mark):.1f}" r="5" fill="#ffd43b"/>')
    parts.append(f'<text x="{X(r_mark)+10:.1f}" y="{Y(h_mark)-8:.1f}" fill="#ffd43b" '
                 f'font-size="11">0.5 mm tube: {h_mark*100:.1f} cm</text>')

    # mark a 1 micron pore (root uptake regime)
    r_pore = 1e-6
    h_pore = capillary_rise(GAMMA_WATER, r_pore)
    parts.append(f'<circle cx="{X(r_pore):.1f}" cy="{Y(h_pore):.1f}" r="5" fill="#06d6a0"/>')
    parts.append(f'<text x="{X(r_pore)+10:.1f}" y="{Y(h_pore)-8:.1f}" fill="#06d6a0" '
                 f'font-size="11">1 um pore: {h_pore:.0f} m</text>')

    parts.append(f'<text x="{x1:.1f}" y="{y1-8:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="end">a slope of -1 on log-log: rise doubles when radius halves</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
