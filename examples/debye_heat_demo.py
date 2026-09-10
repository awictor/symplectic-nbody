"""Demo: the Debye model of solid heat capacity.

Prints the heat capacity of several materials at room temperature and the universal
Debye curve, then draws C_V vs T/Theta_D -- the single curve every solid follows, rising
from the T^3 law to the Dulong-Petit 3R plateau.

    python examples/debye_heat_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from debye_heat import (heat_capacity, low_temperature_limit,  # noqa: E402
                        fraction_of_dulong_petit, DULONG_PETIT, R_GAS,
                        THETA_DIAMOND, THETA_COPPER, THETA_LEAD)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Debye model: C_V rises as T^3, plateaus at Dulong-Petit 3R = "
          f"{DULONG_PETIT:.2f} J/mol/K\n")
    print(f"  {'material':>12}{'Theta_D (K)':>13}{'C_V @300K':>12}{'% of 3R':>10}")
    print("  " + "-" * 48)
    mats = [("lead", THETA_LEAD), ("copper", THETA_COPPER),
            ("aluminium", 428.0), ("diamond", THETA_DIAMOND)]
    for name, th in mats:
        C = heat_capacity(300.0, th)
        print(f"  {name:>12}{th:>13.0f}{C:>12.2f}{fraction_of_dulong_petit(300.0, th)*100:>9.1f}%")

    print("\n  Every solid follows one universal curve in T/Theta_D. A stiff, light")
    print("  lattice has a high Debye temperature, so at room temperature it is still")
    print("  'cold' -- diamond stores only ~1/6 of its classical heat capacity, while")
    print("  soft heavy lead has long since reached the 3R plateau. The T^3 falloff at")
    print("  low temperature is the fingerprint of phonon quantization.")

    _svg(os.path.join(outdir, "debye_heat.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'debye_heat.svg')}")


def _svg(path, size=720, pad=72):
    # universal curve: C_V / 3R vs T/Theta_D
    ratios = [0.02 * i for i in range(1, 81)]   # 0.02 .. 1.6
    theta = 1.0   # work in reduced units: heat_capacity(T, 1) with T = ratio
    cv = [heat_capacity(rr, 1.0) / DULONG_PETIT for rr in ratios]
    xmin, xmax = 0.0, ratios[-1]
    ymin, ymax = 0.0, 1.05

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
    # Dulong-Petit plateau line
    y1 = sy(1.0)
    parts.append(f'<line x1="{pad}" y1="{y1:.1f}" x2="{size-pad}" y2="{y1:.1f}" '
                 f'stroke="#ff6b6b" stroke-width="1.2" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{size-pad-6:.1f}" y="{y1-6:.1f}" fill="#ff6b6b" '
                 f'font-size="11" text-anchor="end">Dulong-Petit 3R</text>')

    # T^3 low-T dashed
    t3 = [low_temperature_limit(rr, 1.0) / DULONG_PETIT for rr in ratios]
    p3 = " ".join(f"{sx(ratios[i]):.1f},{sy(min(t3[i], 1.05)):.1f}" for i in range(len(ratios)))
    parts.append(f'<polyline points="{p3}" fill="none" stroke="#06d6a0" '
                 f'stroke-width="1.4" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{sx(0.5):.1f}" y="{sy(0.15):.1f}" fill="#06d6a0" '
                 f'font-size="10">T^3 law</text>')

    poly = " ".join(f"{sx(ratios[i]):.1f},{sy(cv[i]):.1f}" for i in range(len(ratios)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ffd43b" stroke-width="2.6"/>')

    # mark materials at T=300K on the reduced axis
    for name, th, col in [("diamond", THETA_DIAMOND, "#4dabf7"),
                          ("copper", THETA_COPPER, "#ff922b"),
                          ("lead", THETA_LEAD, "#b197fc")]:
        r = 300.0 / th
        if r <= xmax:
            px = sx(r)
            py = sy(fraction_of_dulong_petit(300.0, th))
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{col}"/>')
            parts.append(f'<text x="{px+6:.1f}" y="{py+4:.1f}" fill="{col}" '
                         f'font-size="10">{name}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'The universal Debye heat-capacity curve</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'C_V / 3R vs T/Theta_D: T^3 rise to the classical plateau (dots at 300 K)</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">T / Theta_D -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'C_V / 3R</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
