"""Demo: the protoplanetary snow line.

Prints the disk temperature at each planet and the frost-line distances of water,
CO2 and CO, then draws the T ~ r^(-1/2) disk-temperature profile with the frost
lines and planet orbits marked -- the rocky-inside / icy-outside divide.

    python examples/snow_line_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from snow_line import (disk_temperature, snow_line_au, L_SUN, AU,  # noqa: E402
                       T_WATER_ICE, T_CO2_ICE, T_CO_ICE)

PLANETS = [("Mercury", 0.39), ("Venus", 0.72), ("Earth", 1.0), ("Mars", 1.52),
           ("Jupiter", 5.20), ("Saturn", 9.58), ("Uranus", 19.2), ("Neptune", 30.1)]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("The snow line: disk T(r) = (L / 16 pi sigma r^2)^(1/4) ~ r^(-1/2)\n")
    print(f"  {'planet':>10}{'r (AU)':>9}{'T (K)':>9}{'state':>10}")
    print("  " + "-" * 38)
    r_snow = snow_line_au()
    for name, r_au in PLANETS:
        T = disk_temperature(r_au * AU)
        state = "rock" if r_au < r_snow else "rock+ice"
        print(f"  {name:>10}{r_au:>9.2f}{T:>9.1f}{state:>10}")

    print(f"\n  water snow line: {r_snow:.2f} AU  (T = {T_WATER_ICE:.0f} K)")
    print(f"  CO2  frost line: {snow_line_au(T_CO2_ICE):.1f} AU")
    print(f"  CO   frost line: {snow_line_au(T_CO_ICE):.0f} AU (out past Neptune)")
    print("\n  Inside ~3 AU water is vapour, so only rock and metal condense and the")
    print("  terrestrial planets grew small and dry. Beyond it, ice roughly triples")
    print("  the solid surface density, letting Jupiter's core grow fast enough to")
    print("  grab nebular gas before the disk dissipated. The snow line is the")
    print("  dividing line between the rocky inner and giant/icy outer solar system.")

    _svg(os.path.join(outdir, "snow_line.svg"), r_snow)
    print(f"\n  wrote {os.path.join(outdir, 'snow_line.svg')}")


def _svg(path, r_snow_au, size=720, pad=66):
    r_au = [0.3 * (1.1 ** i) for i in range(0, 55)]   # 0.3 .. ~40 AU
    r_au = [x for x in r_au if x <= 40.0]
    T = [disk_temperature(x * AU) for x in r_au]
    lx = [math.log10(x) for x in r_au]
    ly = [math.log10(t) for t in T]
    xmin, xmax = lx[0], lx[-1]
    ymin, ymax = min(ly), max(ly)

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    # frost-line vertical markers
    for T_c, label, col in [(T_WATER_ICE, "H2O", "#4dabf7"),
                            (T_CO2_ICE, "CO2", "#06d6a0"),
                            (T_CO_ICE, "CO", "#b197fc")]:
        a = snow_line_au(T_c)
        if r_au[0] <= a <= r_au[-1]:
            cx = sx(math.log10(a))
            parts.append(f'<line x1="{cx:.1f}" y1="{pad}" x2="{cx:.1f}" '
                         f'y2="{size-pad}" stroke="{col}" stroke-width="1.2" '
                         f'stroke-dasharray="4 4"/>')
            parts.append(f'<text x="{cx+3:.1f}" y="{pad+14:.1f}" fill="{col}" '
                         f'font-size="10">{label} ({a:.1f} AU)</text>')

    poly = " ".join(f"{sx(lx[i]):.1f},{sy(ly[i]):.1f}" for i in range(len(r_au)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ff922b" stroke-width="2.4"/>')

    # planet ticks along the bottom
    for name, pa in PLANETS:
        if r_au[0] <= pa <= r_au[-1]:
            px = sx(math.log10(pa))
            parts.append(f'<circle cx="{px:.1f}" cy="{size-pad:.1f}" r="3" fill="#8b949e"/>')
            parts.append(f'<text x="{px:.1f}" y="{size-pad+16:.1f}" fill="#8b949e" '
                         f'font-size="8" text-anchor="middle">{name[:3]}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'The snow line: disk temperature vs distance</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'T ~ r^(-1/2); ice condenses where the disk cools past ~160 K</text>')
    parts.append(f'<text x="{pad-12}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 disk temperature (K)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
