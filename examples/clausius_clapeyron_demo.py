"""Demo: the Clausius-Clapeyron vapor-pressure relation.

Prints water's boiling point at several altitudes and its vapor pressure across
temperature, then draws the exponential vapor-pressure curve with the 1-atm boiling
point and a few altitude markers.

    python examples/clausius_clapeyron_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clausius_clapeyron import (vapor_pressure, boiling_point,  # noqa: E402
                                pressure_from_altitude, P_ATM)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Clausius-Clapeyron: P(T) = P0 exp(-(L/R)(1/T - 1/T0))\n")
    print(f"  water boiling point vs altitude:")
    print(f"  {'location':>18}{'altitude (m)':>14}{'pressure (kPa)':>16}"
          f"{'boils at (C)':>14}")
    print("  " + "-" * 62)
    places = [
        ("sea level", 0), ("Denver", 1609), ("La Paz", 3640),
        ("Everest base", 5364), ("Everest summit", 8848),
    ]
    for name, h in places:
        P = pressure_from_altitude(h)
        Tb = boiling_point(P) - 273.15
        print(f"  {name:>18}{h:>14}{P/1000:>16.1f}{Tb:>14.1f}")

    print("\n  Vapor pressure climbs exponentially with temperature -- roughly doubling")
    print("  every ~15 K -- so water boils when its vapor pressure reaches the ambient")
    print("  air pressure. Up a mountain the thinner air lets it boil cooler (72 C on")
    print("  Everest, too cold to cook an egg), while a pressure cooker raises the")
    print("  boiling point to ~121 C. The same curve sets humidity and cloud formation.")

    _svg(os.path.join(outdir, "clausius_clapeyron.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'clausius_clapeyron.svg')}")


def _svg(path, size=720, pad=72):
    Ts = [273.0 + 2.0 * i for i in range(0, 66)]   # 0 .. 130 C
    Ps = [vapor_pressure(T) / 1000.0 for T in Ts]   # kPa
    xs = [T - 273.15 for T in Ts]                   # deg C
    xmin, xmax = xs[0], xs[-1]
    ymin, ymax = 0.0, max(Ps)

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
    # 1 atm line
    y1 = sy(101.325)
    parts.append(f'<line x1="{pad}" y1="{y1:.1f}" x2="{size-pad}" y2="{y1:.1f}" '
                 f'stroke="#8b949e" stroke-width="1" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{pad+8}" y="{y1-6:.1f}" fill="#8b949e" font-size="10">1 atm -> boils at 100 C</text>')

    poly = " ".join(f"{sx(xs[i]):.1f},{sy(Ps[i]):.1f}" for i in range(len(xs)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')

    for name, h, col in [("Everest", 8848, "#ff6b6b"), ("Denver", 1609, "#ffd43b"),
                         ("cooker 2atm", -1, "#06d6a0")]:
        if h >= 0:
            P = pressure_from_altitude(h) / 1000.0
            Tb = boiling_point(P * 1000.0) - 273.15
        else:
            P = 2 * 101.325
            Tb = boiling_point(P * 1000.0) - 273.15
        if xmin <= Tb <= xmax and ymin <= P <= ymax:
            parts.append(f'<circle cx="{sx(Tb):.1f}" cy="{sy(P):.1f}" r="4.5" fill="{col}"/>')
            parts.append(f'<text x="{sx(Tb)+6:.1f}" y="{sy(P)+4:.1f}" fill="{col}" '
                         f'font-size="10">{name} ({Tb:.0f} C)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Water vapor pressure vs temperature</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'boiling is where the curve meets the ambient pressure</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">temperature (C) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'vapor pressure (kPa)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
