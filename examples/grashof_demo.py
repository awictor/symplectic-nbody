"""Demo: the Grashof number -- convection a hot surface drives itself.

Prints the Grashof/Rayleigh numbers and natural-convection heat-transfer coefficient for
warm surfaces of growing height, then draws h and the flow regime versus wall height: the
buoyant boundary layer strengthening and tripping from laminar to turbulent at Ra ~ 1e9.

    python examples/grashof_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grashof import (expansion_coefficient_ideal_gas, grashof_number,  # noqa: E402
                     rayleigh_number, nusselt_vertical_plate, is_turbulent,
                     heat_transfer_coefficient)


# air properties at ~300 K
BETA = expansion_coefficient_ideal_gas(300.0)
NU = 1.5e-5
PR = 0.71
K = 0.026
DT = 20.0        # 20 K warmer than the room


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Grashof number Gr = g beta dT L^3 / nu^2: buoyancy-driven natural convection\n")
    print("  Vertical wall 20 K above still air (beta=1/300, Pr=0.71):\n")
    print(f"  {'height L':>10}{'Gr':>12}{'Ra':>12}{'regime':>12}{'h (W/m^2K)':>13}")
    for L in (0.05, 0.3, 1.0, 3.0, 10.0):
        Gr = grashof_number(DT, L, BETA, NU)
        Ra = rayleigh_number(Gr, PR)
        Nu = nusselt_vertical_plate(Ra)
        h = heat_transfer_coefficient(Nu, K, L)
        regime = "turbulent" if is_turbulent(Ra) else "laminar"
        print(f"  {L:>8.2f} m{Gr:>12.2e}{Ra:>12.2e}{regime:>12}{h:>13.2f}")

    print("\n  Buoyancy alone gives only a few W/(m^2 K) -- the gentle warmth off a radiator")
    print("  or a sun-baked wall, an order of magnitude below a fan's forced convection. The")
    print("  layer stays laminar until Ra ~ 1e9 (around a metre here), then trips turbulent.")

    _svg(os.path.join(outdir, "grashof.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'grashof.svg')}")


def _svg(path, size=720, pad=80):
    # h vs wall height on log-log, with the laminar/turbulent transition marked.
    Ls = [10 ** (-1.6 + (1.2 - (-1.6)) * i / 199) for i in range(200)]   # 0.025 .. ~16 m
    hs, ras = [], []
    for L in Ls:
        Ra = rayleigh_number(grashof_number(DT, L, BETA, NU), PR)
        ras.append(Ra)
        hs.append(heat_transfer_coefficient(nusselt_vertical_plate(Ra), K, L))

    lx0, lx1 = math.log10(Ls[0]), math.log10(Ls[-1])
    hy0, hy1 = min(hs) * 0.9, max(hs) * 1.1

    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 44

    def X(L):
        return x0 + (math.log10(L) - lx0) / (lx1 - lx0) * (x1 - x0)

    def Y(h):
        return y0 - (h - hy0) / (hy1 - hy0) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Natural convection off a warm wall</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'buoyancy sets the heat-transfer coefficient; the layer trips turbulent at Ra ~ 1e9</text>',
    ]

    # find transition height (Ra crosses 1e9)
    Lt = None
    for i in range(1, len(Ls)):
        if ras[i - 1] < 1e9 <= ras[i]:
            Lt = Ls[i]
            break

    # shade turbulent region
    if Lt:
        parts.append(f'<rect x="{X(Lt):.1f}" y="{y1:.1f}" width="{x1-X(Lt):.1f}" '
                     f'height="{y0-y1:.1f}" fill="#ff6b6b" opacity="0.08"/>')
        parts.append(f'<line x1="{X(Lt):.1f}" y1="{y1:.1f}" x2="{X(Lt):.1f}" y2="{y0:.1f}" '
                     f'stroke="#ff6b6b" stroke-width="1.5" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{X(Lt)+6:.1f}" y="{y1+14:.1f}" fill="#ff6b6b" font-size="11">'
                     f'Ra = 1e9 (L ~ {Lt:.1f} m): laminar -> turbulent</text>')

    # axes
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.5"/>')
    for hval in range(0, int(hy1) + 1, 1):
        if hval < hy0:
            continue
        gy = Y(hval)
        parts.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{x0-8:.1f}" y="{gy+4:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="end">{hval}</text>')
    for e in (-1, 0, 1):
        gx = X(10.0 ** e)
        parts.append(f'<line x1="{gx:.1f}" y1="{y0:.1f}" x2="{gx:.1f}" y2="{y0+4:.1f}" stroke="#8b949e"/>')
        parts.append(f'<text x="{gx:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{"0.1" if e==-1 else ("1" if e==0 else "10")} m</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+30:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">wall height L</text>')
    parts.append(f'<text x="24" y="{(y0+y1)/2:.1f}" fill="#8b949e" font-size="11" '
                 f'transform="rotate(-90 24 {(y0+y1)/2:.1f})" text-anchor="middle">h (W/m^2 K)</text>')

    # h curve
    poly = " ".join(f"{X(Ls[i]):.1f},{Y(hs[i]):.1f}" for i in range(len(Ls)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')
    parts.append(f'<text x="{x0+90:.1f}" y="{Y(hs[0]):.1f}" fill="#4dabf7" font-size="11">'
                 f'h = Nu k / L, 20 K warm wall in still air</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
