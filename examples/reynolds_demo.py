"""Demo: the Reynolds number across the natural world.

Prints Re and the flow regime for systems from a swimming bacterium to a whale, then
draws them on a log Reynolds axis with the laminar/turbulent transition band shaded --
one number spanning 13 orders of magnitude.

    python examples/reynolds_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reynolds import (reynolds_number, flow_regime, critical_velocity,  # noqa: E402
                      RHO_WATER, MU_WATER, RE_LAMINAR, RE_TURBULENT)

RHO_AIR = 1.225
MU_AIR = 1.81e-5


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Reynolds number Re = rho v L / mu: inertia vs viscosity\n")
    print(f"  {'system':>18}{'v (m/s)':>10}{'L (m)':>10}{'Re':>12}{'regime':>14}")
    print("  " + "-" * 64)
    # (name, v, L, rho, mu)
    cases = [
        ("bacterium", 30e-6, 1e-6, RHO_WATER, MU_WATER),
        ("sperm cell", 2e-4, 5e-5, RHO_WATER, MU_WATER),
        ("blood in capillary", 1e-3, 8e-6, RHO_WATER, MU_WATER),
        ("water tap (pipe)", 1.0, 0.02, RHO_WATER, MU_WATER),
        ("swimming human", 1.5, 1.8, RHO_WATER, MU_WATER),
        ("airliner wing", 250.0, 3.0, RHO_AIR, MU_AIR),
        ("blue whale", 10.0, 25.0, RHO_WATER, MU_WATER),
    ]
    for name, v, L, rho, mu in cases:
        Re = reynolds_number(rho, v, L, mu)
        print(f"  {name:>18}{v:>10.2g}{L:>10.2g}{Re:>12.1e}{flow_regime(Re):>14}")

    print("\n  Thirteen orders of magnitude of Re separate a bacterium from a whale.")
    print("  At tiny Re viscosity rules -- a microbe cannot coast, it is like swimming")
    print("  in honey. At huge Re inertia rules and the flow tumbles into turbulence.")
    print("  For pipe flow the crossover sits near Re ~ 2300, which is why a slow tap")
    print("  runs in smooth laminar sheets but a fast one roars and sputters.")

    _svg(os.path.join(outdir, "reynolds.svg"), cases)
    print(f"\n  wrote {os.path.join(outdir, 'reynolds.svg')}")


def _svg(path, cases, size=720, pad=80):
    res = [(name, reynolds_number(rho, v, L, mu)) for name, v, L, rho, mu in cases]
    lx = [math.log10(Re) for _, Re in res]
    xmin, xmax = min(lx) - 1, max(lx) + 1

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    axis_y = size - pad
    parts.append(f'<line x1="{pad}" y1="{axis_y}" x2="{size-pad}" y2="{axis_y}" stroke="#30363d"/>')

    # transition band 2300..4000
    xb0 = sx(math.log10(RE_LAMINAR))
    xb1 = sx(math.log10(RE_TURBULENT))
    parts.append(f'<rect x="{xb0:.1f}" y="{pad}" width="{xb1-xb0:.1f}" '
                 f'height="{axis_y-pad:.1f}" fill="#ff6b6b" fill-opacity="0.15"/>')
    parts.append(f'<text x="{(xb0+xb1)/2:.1f}" y="{pad+18:.1f}" fill="#ff6b6b" '
                 f'font-size="10" text-anchor="middle">transition</text>')
    parts.append(f'<text x="{xb0-10:.1f}" y="{pad+34:.1f}" fill="#4dabf7" '
                 f'font-size="11" text-anchor="end">laminar</text>')
    parts.append(f'<text x="{xb1+10:.1f}" y="{pad+34:.1f}" fill="#ffd43b" '
                 f'font-size="11">turbulent</text>')

    # plot each system as a labelled dot, staggered vertically
    for i, (name, Re) in enumerate(res):
        x = sx(math.log10(Re))
        y = pad + 60 + i * ((axis_y - pad - 80) / max(1, len(res) - 1))
        col = "#4dabf7" if Re < RE_LAMINAR else ("#ffd43b" if Re > RE_TURBULENT else "#ff6b6b")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{col}"/>')
        parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{axis_y:.1f}" '
                     f'stroke="{col}" stroke-width="0.6" stroke-dasharray="2 3" opacity="0.5"/>')
        parts.append(f'<text x="{x+7:.1f}" y="{y+4:.1f}" fill="{col}" '
                     f'font-size="10">{name} (Re {Re:.0e})</text>')

    parts.append(f'<text x="20" y="34" fill="#e6edf3" font-size="18">'
                 f'Reynolds number across the natural world</text>')
    parts.append(f'<text x="20" y="52" fill="#8b949e" font-size="12">'
                 f'13 decades from a bacterium to a whale; pipe transition near Re ~ 2300</text>')
    parts.append(f'<text x="{size-pad}" y="{axis_y+22:.1f}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 Reynolds number -&gt;</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
