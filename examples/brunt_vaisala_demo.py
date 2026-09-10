"""Demo: the Brunt-Vaisala frequency and convective stability.

Prints N and the buoyancy period for several atmospheric layers, then draws N^2 vs
the environmental lapse rate, shading the convectively-unstable region beyond the
adiabatic lapse rate (the Schwarzschild criterion).

    python examples/brunt_vaisala_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from brunt_vaisala import (adiabatic_lapse_rate, brunt_vaisala_squared_ideal,  # noqa: E402
                           brunt_vaisala_frequency, buoyancy_period,
                           is_convective, G_EARTH, CP_AIR)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    lapse = adiabatic_lapse_rate() * 1000.0
    print("Brunt-Vaisala buoyancy frequency: N^2 = (g/T)(dT/dz + g/c_p)\n")
    print(f"  dry adiabatic lapse rate g/c_p = {lapse:.2f} K/km "
          f"(the stability threshold)\n")
    print(f"  {'layer':>22}{'dT/dz (K/km)':>14}{'N (1/s)':>10}"
          f"{'period':>12}{'state':>13}")
    print("  " + "-" * 71)
    layers = [
        ("strong inversion", +10.0, 273.0),
        ("stratosphere", +2.0, 220.0),
        ("isothermal", 0.0, 273.0),
        ("typical troposphere", -6.5, 273.0),
        ("dry adiabatic", -9.76, 273.0),
        ("superadiabatic", -15.0, 300.0),
    ]
    for name, dTdz_km, T in layers:
        N2 = brunt_vaisala_squared_ideal(dTdz_km / 1000.0, T)
        N = brunt_vaisala_frequency(N2)
        if N > 0:
            per = f"{buoyancy_period(N2)/60.0:.1f} min"
            state = "stable"
        else:
            per = "-- "
            state = "CONVECTIVE"
        print(f"  {name:>22}{dTdz_km:>14.1f}{N:>10.4f}{per:>12}{state:>13}")

    print("\n  A parcel displaced in a stable layer overshoots and oscillates at N --")
    print("  these buoyancy (internal gravity) waves ripple through the stratosphere")
    print("  and the Sun's radiative core. Once the environment cools faster than the")
    print("  adiabatic lapse rate, N^2 flips negative: buoyancy runs away and the")
    print("  layer convects. That sign change IS the Schwarzschild convection criterion.")

    _svg(os.path.join(outdir, "brunt_vaisala.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'brunt_vaisala.svg')}")


def _svg(path, size=720, pad=70, T=273.0):
    # x = environmental lapse rate -dT/dz (K/km): larger = steeper cooling upward
    lapses_km = [-4.0 + 0.2 * i for i in range(0, 96)]   # -4 .. +15 K/km cooling
    N2 = [brunt_vaisala_squared_ideal(-L / 1000.0, T) for L in lapses_km]
    xmin, xmax = lapses_km[0], lapses_km[-1]
    ymax = max(N2) * 1.1
    ymin = min(N2) * 1.1

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    adiab = adiabatic_lapse_rate() * 1000.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # convective region: lapse > adiabatic -> N^2 < 0. Shade to the right of adiab.
    xa = sx(adiab)
    parts.append(f'<rect x="{xa:.1f}" y="{pad}" width="{size-pad-xa:.1f}" '
                 f'height="{size-2*pad:.1f}" fill="#ff6b6b" fill-opacity="0.12"/>')
    parts.append(f'<line x1="{xa:.1f}" y1="{pad}" x2="{xa:.1f}" y2="{size-pad}" '
                 f'stroke="#ff6b6b" stroke-width="1.4" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{xa+6:.1f}" y="{pad+18:.1f}" fill="#ff6b6b" '
                 f'font-size="11">convective (lapse &gt; {adiab:.1f} K/km)</text>')

    # N^2 = 0 axis
    y0 = sy(0.0)
    parts.append(f'<line x1="{pad}" y1="{y0:.1f}" x2="{size-pad}" y2="{y0:.1f}" stroke="#30363d"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>')
    parts.append(f'<text x="{pad+6:.1f}" y="{y0-6:.1f}" fill="#8b949e" font-size="10">N^2 = 0</text>')

    poly = " ".join(f"{sx(lapses_km[i]):.1f},{sy(N2[i]):.1f}" for i in range(len(lapses_km)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Buoyancy frequency squared vs lapse rate</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'N^2 &gt; 0 stable (gravity waves); N^2 &lt; 0 convects (Schwarzschild)</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">environmental lapse rate (K/km) -&gt;</text>')
    parts.append(f'<text x="{pad-16}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'N^2 (s^-2)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
