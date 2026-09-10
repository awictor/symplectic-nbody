"""Demo: the Joule-Thomson effect and gas liquefaction.

Prints the inversion temperature and room-temperature behaviour of common gases, then
draws the JT coefficient vs temperature, crossing zero at each gas's inversion
temperature -- below it throttling cools (liquefies), above it warms.

    python examples/joule_thomson_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from joule_thomson import (jt_coefficient, inversion_temperature,  # noqa: E402
                           cools_on_expansion, temperature_change,
                           A_N2, B_N2, CP_N2, A_H2, B_H2, CP_H2,
                           A_HE, B_HE, CP_HE, A_CO2, B_CO2, CP_CO2)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Joule-Thomson: throttling cools below T_inv, warms above\n")
    print(f"  {'gas':>8}{'T_inv (K)':>12}{'at 300 K':>14}{'mu (K/MPa) @300K':>18}")
    print("  " + "-" * 52)
    gases = [("CO2", A_CO2, B_CO2, CP_CO2), ("N2", A_N2, B_N2, CP_N2),
             ("H2", A_H2, B_H2, CP_H2), ("He", A_HE, B_HE, CP_HE)]
    for name, a, b, cp in gases:
        tinv = inversion_temperature(a, b)
        state = "cools" if cools_on_expansion(300.0, a, b) else "warms"
        mu = jt_coefficient(300.0, a, b, cp) * 1e6
        print(f"  {name:>8}{tinv:>12.0f}{state:>14}{mu:>18.3f}")

    print("\n  Two effects compete when a real gas expands through a valve: attraction")
    print("  cools it (work pulling molecules apart), finite size warms it. Below the")
    print("  inversion temperature attraction wins, so nitrogen and CO2 cool at room")
    print("  temperature and can be liquefied by repeated throttling. Hydrogen and")
    print("  helium have low inversion temperatures, so they must be pre-cooled first")
    print("  -- otherwise throttling heats them, an early liquefaction hazard.")

    _svg(os.path.join(outdir, "joule_thomson.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'joule_thomson.svg')}")


def _svg(path, size=720, pad=72):
    Ts = [20.0 + 15.0 * i for i in range(0, 70)]   # 20 .. ~1055 K
    curves = [("N2", A_N2, B_N2, CP_N2, "#4dabf7"),
              ("H2", A_H2, B_H2, CP_H2, "#ffd43b"),
              ("He", A_HE, B_HE, CP_HE, "#ff6b6b")]
    data = []
    for name, a, b, cp, col in curves:
        ys = [jt_coefficient(T, a, b, cp) * 1e6 for T in Ts]   # K/MPa
        data.append((name, col, ys, a, b))

    xmin, xmax = Ts[0], Ts[-1]
    allv = [y for _, _, ys, _, _ in data for y in ys]
    ymin, ymax = min(allv), max(allv)

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
    # mu = 0 line (inversion)
    y0 = sy(0.0)
    parts.append(f'<line x1="{pad}" y1="{y0:.1f}" x2="{size-pad}" y2="{y0:.1f}" '
                 f'stroke="#8b949e" stroke-width="1" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{pad+8}" y="{y0-6:.1f}" fill="#8b949e" '
                 f'font-size="10">mu = 0: below cools, above warms</text>')

    ytop = pad + 22
    for i, (name, col, ys, a, b) in enumerate(data):
        pts = " ".join(f"{sx(Ts[j]):.1f},{sy(ys[j]):.1f}" for j in range(len(Ts)))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        # inversion crossing tick
        tinv = inversion_temperature(a, b)
        if xmin <= tinv <= xmax:
            parts.append(f'<circle cx="{sx(tinv):.1f}" cy="{y0:.1f}" r="3.5" fill="{col}"/>')
        parts.append(f'<text x="{pad+10}" y="{ytop + i*16}" fill="{col}" '
                     f'font-size="12">{name} (T_inv={tinv:.0f} K)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Joule-Thomson coefficient vs temperature</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'mu > 0 cools on throttling; the zero crossing is the inversion temperature</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">temperature (K) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'mu_JT (K / MPa)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
