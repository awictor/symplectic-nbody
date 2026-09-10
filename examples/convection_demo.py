"""Demo: convective heat transfer -- how fast a fluid carries heat off a surface.

Prints the heat-transfer coefficient and Biot number across regimes (still air to forced
water) and draws a hot aluminium block cooling from 100 C: exponential curves whose time
constant shrinks as the convection stiffens from still air to a breeze to forced water.

    python examples/convection_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from convection import (dittus_boelter, heat_transfer_coefficient, biot_number,  # noqa: E402
                        lumped_time_constant, lumped_temperature)


# a 1 cm aluminium cube cooling in various fluids
RHO_AL, CP_AL, K_AL = 2700.0, 900.0, 205.0
SIDE = 0.01
VOL = SIDE ** 3
AREA = 6 * SIDE ** 2
T0, TINF = 100.0, 20.0

REGIMES = [
    ("still air", 8.0, "#4dabf7"),
    ("breeze / fan", 40.0, "#06d6a0"),
    ("forced water", 2000.0, "#ff6b6b"),
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Convective heat transfer: q = h (T_s - T_inf),  Nu = h L / k\n")
    print("  Forced water in a pipe (Dittus-Boelter Nu = 0.023 Re^0.8 Pr^0.4):")
    nu = dittus_boelter(2e4, 7.0, heating=True)
    h_pipe = heat_transfer_coefficient(nu, 0.6, 0.02)
    print(f"    Re=2e4, Pr=7  ->  Nu = {nu:.0f},  h = {h_pipe:.0f} W/(m^2 K)\n")

    print("  1 cm aluminium cube cooling from 100 C in 20 C surroundings:")
    print(f"  {'regime':<16}{'h (W/m^2K)':>12}{'Biot':>10}{'tau':>12}{'t to 30 C':>12}")
    for label, h, _ in REGIMES:
        Bi = biot_number(h, SIDE / 2.0, K_AL)
        tau = lumped_time_constant(RHO_AL, CP_AL, VOL, h, AREA)
        # time to reach 30 C: t = -tau ln((30-Tinf)/(T0-Tinf))
        t30 = -tau * math.log((30.0 - TINF) / (T0 - TINF))
        print(f"  {label:<16}{h:>12.0f}{Bi:>10.4f}{_fmt(tau):>12}{_fmt(t30):>12}")

    print("\n  All three stay near-isothermal inside (Bi << 1), so cooling is a clean")
    print("  exponential T(t) = T_inf + (T0 - T_inf) e^(-t/tau). Stiffer convection = bigger h")
    print("  = shorter tau: still air takes ~an hour, forced water seconds. Newton's law again.")

    _svg(os.path.join(outdir, "convection.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'convection.svg')}")


def _fmt(t):
    if t < 90:
        return f"{t:.1f} s"
    if t < 5400:
        return f"{t/60:.1f} min"
    return f"{t/3600:.1f} hr"


def _svg(path, size=720, pad=76):
    # cooling curves T(t) for each regime; x-axis in seconds on a shared window
    taus = [(label, lumped_time_constant(RHO_AL, CP_AL, VOL, h, AREA), col)
            for label, h, col in REGIMES]
    t_max = 3.0 * max(tau for _, tau, _ in taus)     # window from the slowest (still air)

    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 40

    def X(t):
        return x0 + t / t_max * (x1 - x0)

    def Y(T):
        return y0 - (T - TINF) / (T0 - TINF) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Newtonian cooling of a hot block</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'stiffer convection (bigger h) = shorter time constant; T = T_inf + (T0-T_inf) e^(-t/tau)</text>',
    ]

    # axes
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.5"/>')

    # temperature gridlines
    for T in (20, 40, 60, 80, 100):
        gy = Y(T)
        parts.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{x0-8:.1f}" y="{gy+4:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="end">{T} C</text>')
    # ambient line
    parts.append(f'<text x="{x1-4:.1f}" y="{Y(TINF)-6:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end">ambient 20 C</text>')

    n = 240
    for label, tau, col in taus:
        pts = []
        for k in range(n + 1):
            t = t_max * k / n
            pts.append(f"{X(t):.1f},{Y(lumped_temperature(t, T0, TINF, tau)):.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2.6"/>')
        # mark one time constant
        gx, gy = X(tau), Y(lumped_temperature(tau, T0, TINF, tau))
        parts.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="4" fill="{col}"/>')
        parts.append(f'<text x="{gx+7:.1f}" y="{gy-6:.1f}" fill="{col}" font-size="11">'
                     f'{label} (tau={_fmt(tau)})</text>')

    parts.append(f'<text x="{x1:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end">time (window = 3x the still-air tau)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
