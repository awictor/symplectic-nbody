"""Demo: the Stefan problem -- a freezing front advancing as sqrt(t).

Prints how deep ice grows over time for several frost severities and the Stefan growth
coefficient lambda, then draws ice thickness vs time -- the sqrt(t) curves that show thin
ice forming fast and thick ice ever more slowly, with the classic '10 cm in a hard-frost
day' point marked.

    python examples/stefan_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stefan import (stefan_number, solve_lambda, thermal_diffusivity,  # noqa: E402
                    front_position, time_to_depth)


# ice on water
K, RHO, CP, L = 2.2, 917.0, 2100.0, 334000.0
ALPHA = thermal_diffusivity(K, RHO, CP)

FROSTS = [
    ("light frost (-5 C)", 5.0, "#4dabf7"),
    ("hard frost (-15 C)", 15.0, "#06d6a0"),
    ("arctic (-40 C)", 40.0, "#b197fc"),
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Stefan problem: freezing front X(t) = 2 lambda sqrt(alpha t)\n")
    print(f"  ice: alpha = {ALPHA:.2e} m^2/s,  latent heat L = {L/1000:.0f} kJ/kg\n")
    print(f"  {'frost':<22}{'St':>8}{'lambda':>9}{'ice @ 1 day':>14}{'ice @ 1 wk':>13}")
    for label, dT, _ in FROSTS:
        St = stefan_number(CP, dT, L)
        lam = solve_lambda(St)
        x1 = front_position(86400.0, lam, ALPHA)
        x7 = front_position(7 * 86400.0, lam, ALPHA)
        print(f"  {label:<22}{St:>8.3f}{lam:>9.3f}{x1*100:>11.1f} cm{x7*100:>10.1f} cm")

    print("\n  To reach a given thickness (hard frost, -15 C):")
    St = stefan_number(CP, 15.0, L)
    lam = solve_lambda(St)
    for depth_cm in (5, 10, 30, 50):
        t = time_to_depth(depth_cm / 100.0, lam, ALPHA)
        print(f"    {depth_cm:>3} cm  ->  {t/86400:.1f} days")

    print("\n  The front slows as sqrt(t): the latent heat released at the interface must")
    print("  conduct out through the ice already formed, and that layer thickens, so the")
    print("  first few cm come in hours but the next foot takes weeks. Same law crusts lava.")

    _svg(os.path.join(outdir, "stefan.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'stefan.svg')}")


def _svg(path, size=720, pad=78):
    t_max = 14 * 86400.0                 # two weeks
    curves = []
    xmax = 0.0
    for label, dT, col in FROSTS:
        lam = solve_lambda(stefan_number(CP, dT, L))
        curves.append((label, lam, col))
        xmax = max(xmax, front_position(t_max, lam, ALPHA))
    xmax *= 1.05

    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 40

    def X(t):
        return x0 + t / t_max * (x1 - x0)

    def Y(x):
        return y0 - x / xmax * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Ice growth: the Stefan sqrt(t) front</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'the freezing front slows as it deepens -- latent heat must conduct out through the ice</text>',
    ]

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.5"/>')

    # depth gridlines (cm)
    for xc in range(0, int(xmax * 100) + 1, 10):
        gy = Y(xc / 100.0)
        parts.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{x0-8:.1f}" y="{gy+4:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="end">{xc} cm</text>')
    # time gridlines (days)
    for d in range(0, 15, 2):
        gx = X(d * 86400.0)
        parts.append(f'<line x1="{gx:.1f}" y1="{y0:.1f}" x2="{gx:.1f}" y2="{y0+4:.1f}" stroke="#8b949e"/>')
        parts.append(f'<text x="{gx:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{d}d</text>')

    n = 200
    for label, lam, col in curves:
        pts = []
        for k in range(1, n + 1):
            t = t_max * k / n
            pts.append(f"{X(t):.1f},{Y(front_position(t, lam, ALPHA)):.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2.6"/>')
        xend = front_position(t_max, lam, ALPHA)
        parts.append(f'<text x="{x1-4:.1f}" y="{Y(xend)+4:.1f}" fill="{col}" font-size="11" '
                     f'text-anchor="end">{label}</text>')

    # mark the ~10 cm / 1 day hard-frost point
    lam_hard = solve_lambda(stefan_number(CP, 15.0, L))
    xh = front_position(86400.0, lam_hard, ALPHA)
    parts.append(f'<circle cx="{X(86400.0):.1f}" cy="{Y(xh):.1f}" r="5" fill="#ffd43b"/>')
    parts.append(f'<text x="{X(86400.0)+9:.1f}" y="{Y(xh)+4:.1f}" fill="#ffd43b" font-size="11">'
                 f'~{xh*100:.0f} cm after 1 day of hard frost</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
