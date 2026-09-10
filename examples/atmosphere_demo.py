"""Demo: atmospheric escape -- which worlds keep which gases.

Computes the Jeans escape parameter for several bodies and gases and shows the
retention verdict, then renders escape speed vs thermal speed with the lambda~36
retention boundary to SVG.

    python examples/atmosphere_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from atmosphere import (thermal_speed, escape_speed, escape_parameter,  # noqa: E402
                        is_retained)

# (name, mass kg, radius m, exosphere T K)
BODIES = [
    ("Moon", 7.342e22, 1.737e6, 400),
    ("Mars", 6.417e23, 3.390e6, 300),
    ("Earth", 5.972e24, 6.371e6, 1000),
    ("Jupiter", 1.898e27, 6.991e7, 1000),
]
GASES = [("H2", 2), ("He", 4), ("H2O", 18), ("N2", 28), ("CO2", 44)]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Atmospheric escape: which worlds keep which gases (lambda >= 36 = keep)\n")
    header = "  " + f"{'body':<10}" + "".join(f"{g:>8}" for g, _ in GASES)
    print(header)
    print("  " + "-" * (10 + 8 * len(GASES)))
    for name, M, R, T in BODIES:
        row = f"  {name:<10}"
        for g, amu in GASES:
            keep = "keep" if is_retained(M, R, T, amu) else "lose"
            row += f"{keep:>8}"
        print(row)
    print("\n  Earth loses H2/He but keeps N2/O2/CO2; the Moon and Mars lose the")
    print("  light gases; Jupiter keeps everything. Retention needs the escape")
    print("  speed to beat ~6x the molecules' thermal speed.")

    _svg(os.path.join(outdir, "atmosphere.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'atmosphere.svg')}")


def _svg(path, size=720, pad=64):
    # plot each (body, gas) as a point: x=thermal speed, y=escape speed (km/s)
    pts = []
    for name, M, R, T in BODIES:
        ve = escape_speed(M, R) / 1e3
        for g, amu in GASES:
            vth = thermal_speed(T, amu) / 1e3
            pts.append((vth, ve, is_retained(M, R, T, amu)))
    vth_max = max(p[0] for p in pts) * 1.1
    ve_max = max(p[1] for p in pts) * 1.1

    def sx(vth):
        return pad + vth / vth_max * (size - 2 * pad)

    def sy(ve):
        return size - pad - ve / ve_max * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    # retention boundary: v_esc = 6 v_th (lambda = 36)
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" '
                 f'x2="{sx(ve_max/6):.1f}" y2="{sy(ve_max):.1f}" '
                 f'stroke="#e9c46a" stroke-dasharray="5,4"/>')
    parts.append(f'<text x="{sx(ve_max/6)-4:.1f}" y="{sy(ve_max)+16:.1f}" fill="#e9c46a" '
                 f'font-size="11" text-anchor="end">retention boundary (v_esc = 6 v_th)</text>')
    for vth, ve, keep in pts:
        col = "#4cc9f0" if keep else "#e63946"
        parts.append(f'<circle cx="{sx(vth):.1f}" cy="{sy(ve):.1f}" r="4" fill="{col}"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Atmospheric retention: escape vs thermal speed</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+8}" fill="#4cc9f0" font-size="12">retained</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+26}" fill="#e63946" font-size="12">lost</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">molecular thermal speed (km/s) -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'escape speed (km/s)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
