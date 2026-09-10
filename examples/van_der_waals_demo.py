"""Demo: the van der Waals gas and its isotherms.

Prints critical constants for several gases and the universal 3/8 compressibility, then
draws reduced-variable isotherms -- above the critical temperature they are smooth, below
it they develop the tell-tale van der Waals loop where gas condenses to liquid.

    python examples/van_der_waals_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from van_der_waals import (critical_temperature, critical_pressure,  # noqa: E402
                           critical_volume, critical_compressibility,
                           reduced_pressure, A_CO2, B_CO2, A_WATER, B_WATER,
                           A_HELIUM, B_HELIUM)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Van der Waals: (P + a n^2/V^2)(V - nb) = nRT -- a gas that can condense\n")
    print(f"  {'gas':>10}{'T_c (K)':>10}{'P_c (MPa)':>12}{'Pc Vc / R Tc':>15}")
    print("  " + "-" * 48)
    gases = [("helium", A_HELIUM, B_HELIUM), ("CO2", A_CO2, B_CO2),
             ("water", A_WATER, B_WATER)]
    for name, a, b in gases:
        print(f"  {name:>10}{critical_temperature(a, b):>10.1f}"
              f"{critical_pressure(a, b)/1e6:>12.2f}"
              f"{critical_compressibility(a, b):>15.4f}")

    print("\n  The finite-size (b) and attraction (a) terms give the ideal gas something")
    print("  it lacks: a liquid-vapour transition. Above T_c the isotherm is smooth, but")
    print("  below it a wiggle appears -- pressure would rise with volume, which is")
    print("  unstable, so the gas condenses across it. At the critical point that wiggle")
    print("  is an inflection, and Pc Vc / R Tc = 3/8 for EVERY van der Waals gas.")

    _svg(os.path.join(outdir, "van_der_waals.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'van_der_waals.svg')}")


def _svg(path, size=720, pad=72):
    # reduced isotherms Pr(Vr) for several Tr
    Vrs = [0.5 + 0.02 * i for i in range(0, 176)]   # Vr 0.5 .. 4.0
    temps = [(1.15, "#ff6b6b", "Tr = 1.15 (gas)"),
             (1.0, "#ffd43b", "Tr = 1.00 (critical)"),
             (0.90, "#4dabf7", "Tr = 0.90 (loop)"),
             (0.85, "#06d6a0", "Tr = 0.85")]
    data = []
    for Tr, col, label in temps:
        ys = [reduced_pressure(0, Vr, Tr) for Vr in Vrs]
        data.append((ys, col, label))

    xmin, xmax = Vrs[0], Vrs[-1]
    ymin, ymax = 0.0, 2.0

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (min(max(y, ymin), ymax) - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    # critical point at (1,1)
    parts.append(f'<circle cx="{sx(1.0):.1f}" cy="{sy(1.0):.1f}" r="4" fill="#ffd43b"/>')
    parts.append(f'<text x="{sx(1.0)+6:.1f}" y="{sy(1.0)-6:.1f}" fill="#ffd43b" '
                 f'font-size="10">critical point</text>')

    ytop = pad + 22
    for i, (ys, col, label) in enumerate(data):
        pts = " ".join(f"{sx(Vrs[j]):.1f},{sy(ys[j]):.1f}" for j in range(len(Vrs)))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        parts.append(f'<text x="{size-pad-160}" y="{ytop + i*16}" fill="{col}" '
                     f'font-size="12">{label}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Van der Waals reduced isotherms</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'below T_c the loop appears -- where the gas condenses to liquid</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">reduced volume V/V_c -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'reduced pressure P/P_c</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
