"""Demo: tidal heating -- why Io erupts and Europa has an ocean.

Computes Io's tidal heating and surface flux, compares the Galilean moons, and
shows the steep e^2 dependence that ties the heating to the Laplace resonance.
Renders heating vs eccentricity to SVG.

    python examples/tidal_heating_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tidal_heating import (tidal_heating_rate, io_heating, surface_heat_flux,  # noqa: E402
                           M_JUP, IO_R, IO_A, IO_E, IO_K2_OVER_Q, IO_AREA)

import math

# Galilean moons: (name, radius m, semi-major axis m, eccentricity)
MOONS = [
    ("Io", IO_R, IO_A, IO_E),
    ("Europa", 1.5608e6, 6.711e8, 0.009),
    ("Ganymede", 2.6341e6, 1.0704e9, 0.0013),
    ("Callisto", 2.4103e6, 1.8827e9, 0.0074),
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Tidal heating of the Galilean moons (k2/Q ~ 0.015)\n")
    print(f"  {'moon':<10}{'power (W)':>12}{'flux (W/m^2)':>14}")
    print("  " + "-" * 36)
    for name, R, a, e in MOONS:
        p = tidal_heating_rate(M_JUP, R, a, e, IO_K2_OVER_Q)
        area = 4.0 * math.pi * R ** 2
        print(f"  {name:<10}{p:>12.2e}{surface_heat_flux(p, area):>14.3f}")
    print(f"\n  Io: {io_heating():.1e} W ~ 40x Earth's internal heat flux, which is why")
    print("  it is the most volcanic body known. The heating goes as e^2, and Io's")
    print("  eccentricity is forced by the Laplace 4:2:1 resonance -- no resonance,")
    print("  no eccentricity, no volcanoes.")

    _svg(os.path.join(outdir, "tidal_heating.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'tidal_heating.svg')}")


def _svg(path, size=720, pad=64):
    es = [0.0005 * i for i in range(0, 41)]  # 0 .. 0.02
    powers = [tidal_heating_rate(M_JUP, IO_R, IO_A, e, IO_K2_OVER_Q) / 1e14 for e in es]
    emax = es[-1]
    pmax = max(powers) * 1.1

    def sx(e):
        return pad + e / emax * (size - 2 * pad)

    def sy(p):
        return size - pad - p / pmax * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(es[i]):.1f},{sy(powers[i]):.1f}" for i in range(len(es)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#f4a261" stroke-width="2.2"/>')
    # mark Io's actual eccentricity
    parts.append(f'<line x1="{sx(IO_E):.1f}" y1="{pad}" x2="{sx(IO_E):.1f}" y2="{size-pad}" '
                 f'stroke="#e63946" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{sx(IO_E)+6:.1f}" y="{pad+16}" fill="#e63946" font-size="12">'
                 f'Io (e={IO_E})</text>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Tidal heating vs eccentricity (Io parameters)</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'orbital eccentricity; power ~ e^2 (units of 1e14 W)</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'heating (1e14 W)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
