"""Demo: the habitable zone -- where a planet can hold liquid water.

Shows the equilibrium temperature and habitable-zone bounds for stars of
different luminosity, and renders the inner/outer HZ edges vs stellar luminosity
to a log-log SVG (the zone marches outward as sqrt(L)).

    python examples/habitable_zone_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from habitable_zone import (equilibrium_temperature, habitable_zone,  # noqa: E402
                            hz_center)
from main_sequence import luminosity  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("The habitable zone: liquid-water orbits around a star\n")
    print(f"  Earth's equilibrium temperature: {equilibrium_temperature(1, 1, 0.3):.0f} K")
    print(f"  (greenhouse warms the surface to ~288 K)\n")
    print(f"  {'star':<18}{'L (L_sun)':>11}{'HZ inner':>11}{'HZ outer':>11}")
    print("  " + "-" * 51)
    # (name, mass in Msun) -> L from the main-sequence relation
    stars = [("red dwarf (0.3)", 0.3), ("Sun (1.0)", 1.0),
             ("F star (1.5)", 1.5), ("A star (2.0)", 2.0)]
    for name, M in stars:
        L = luminosity(M)
        inner, outer = habitable_zone(L)
        print(f"  {name:<18}{L:>11.2f}{inner:>9.2f} AU{outer:>8.2f} AU")
    print("\n  The zone marches out as sqrt(L): red-dwarf HZs hug the star (and risk")
    print("  tidal locking), while luminous stars push it far out. This is the target")
    print("  band for finding worlds that could have surface oceans.")

    _svg(os.path.join(outdir, "habitable_zone.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'habitable_zone.svg')}")


def _svg(path, size=720, pad=64):
    Ls = [10 ** (-2 + 0.05 * i) for i in range(0, 81)]  # 0.01 .. 100 L_sun
    inner = [habitable_zone(L)[0] for L in Ls]
    outer = [habitable_zone(L)[1] for L in Ls]
    lL = [math.log10(L) for L in Ls]
    li = [math.log10(d) for d in inner]
    lo = [math.log10(d) for d in outer]
    Lmin, Lmax = lL[0], lL[-1]
    ymin, ymax = min(li), max(lo)

    def sx(x):
        return pad + (x - Lmin) / (Lmax - Lmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    # shade the HZ band
    band = ("".join(f"{sx(lL[i]):.1f},{sy(li[i]):.1f} " for i in range(len(Ls)))
            + "".join(f"{sx(lL[i]):.1f},{sy(lo[i]):.1f} " for i in range(len(Ls) - 1, -1, -1)))
    parts.append(f'<polygon points="{band}" fill="#2a9d8f33" stroke="none"/>')
    ip = " ".join(f"{sx(lL[i]):.1f},{sy(li[i]):.1f}" for i in range(len(Ls)))
    op = " ".join(f"{sx(lL[i]):.1f},{sy(lo[i]):.1f}" for i in range(len(Ls)))
    parts.append(f'<polyline points="{ip}" fill="none" stroke="#e63946" stroke-width="1.8"/>')
    parts.append(f'<polyline points="{op}" fill="none" stroke="#4cc9f0" stroke-width="1.8"/>')
    # mark Earth (Sun, 1 AU)
    parts.append(f'<circle cx="{sx(0):.1f}" cy="{sy(0):.1f}" r="4" fill="#e9c46a"/>')
    parts.append(f'<text x="{sx(0)+8:.1f}" y="{sy(0):.1f}" fill="#e9c46a" font-size="11">Earth</text>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Habitable zone vs stellar luminosity (log-log)</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+8}" fill="#4cc9f0" font-size="12">outer edge</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+26}" fill="#e63946" font-size="12">inner edge (green band = HZ)</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 L / L_sun -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 orbital distance (AU)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
