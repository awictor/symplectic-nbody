"""Demo: the Carnot cycle and the limits of heat engines.

Prints the Carnot efficiency of real engines and the coefficient of performance of
fridges and heat pumps, then draws efficiency vs the temperature ratio T_c/T_h -- the
second-law ceiling every engine runs beneath.

    python examples/carnot_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from carnot import (carnot_efficiency, work_output, heat_rejected,  # noqa: E402
                    cop_refrigerator, cop_heat_pump)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Carnot: no engine beats eta = 1 - T_c/T_h\n")
    print(f"  {'engine':>22}{'T_hot (K)':>11}{'T_cold (K)':>12}{'max eta':>10}")
    print("  " + "-" * 55)
    engines = [
        ("car engine", 2000, 300),
        ("steam power plant", 800, 300),
        ("geothermal", 450, 300),
        ("ocean thermal (OTEC)", 298, 278),
    ]
    for name, Th, Tc in engines:
        print(f"  {name:>22}{Th:>11}{Tc:>12}{carnot_efficiency(Th, Tc):>10.3f}")

    print("\n  running the cycle backwards (T_hot=293 K room, T_cold=273 K):")
    print(f"    refrigerator COP = {cop_refrigerator(293, 273):.1f} "
          f"(heat removed per unit work)")
    print(f"    heat pump COP    = {cop_heat_pump(293, 273):.1f} "
          f"(heat delivered per unit work)")

    print("\n  Some heat must always be dumped to the cold reservoir, so no engine hits")
    print("  100% -- a steam plant at 800 K exhausting to 300 K is capped at 62%, and")
    print("  real losses cut it to ~40%. Reversed, the same cycle is a heat pump that")
    print("  delivers many times the heat of the work it draws (COP >> 1), which is why")
    print("  heat pumps beat resistive heaters -- they move heat rather than make it.")

    _svg(os.path.join(outdir, "carnot.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'carnot.svg')}")


def _svg(path, size=720, pad=72):
    ratios = [0.01 * i for i in range(1, 100)]   # T_c/T_h from 0.01 to 0.99
    etas = [1.0 - r for r in ratios]
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0

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
    poly = " ".join(f"{sx(ratios[i]):.1f},{sy(etas[i]):.1f}" for i in range(len(ratios)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ffd43b" stroke-width="2.6"/>')

    # forbidden region above the curve
    parts.append(f'<text x="{sx(0.55):.1f}" y="{sy(0.75):.1f}" fill="#ff6b6b" '
                 f'font-size="12">forbidden (2nd law)</text>')
    parts.append(f'<text x="{sx(0.1):.1f}" y="{sy(0.2):.1f}" fill="#8b949e" '
                 f'font-size="12">achievable</text>')

    for name, Th, Tc, col in [("car", 2000, 300, "#4dabf7"),
                              ("steam plant", 800, 300, "#06d6a0"),
                              ("OTEC", 298, 278, "#ff922b")]:
        r = Tc / Th
        px = sx(r)
        py = sy(carnot_efficiency(Th, Tc))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{col}"/>')
        parts.append(f'<text x="{px+6:.1f}" y="{py-6:.1f}" fill="{col}" '
                     f'font-size="10">{name}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Carnot efficiency vs reservoir temperature ratio</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'eta = 1 - T_c/T_h: the ceiling every real engine sits under</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">T_cold / T_hot -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'maximum efficiency</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
