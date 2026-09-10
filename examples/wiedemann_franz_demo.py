"""Demo: the Wiedemann-Franz law -- heat and charge carried by the same electrons.

Prints the thermal conductivity Wiedemann-Franz predicts for real metals from their
electrical conductivity, next to the measured value and the effective Lorenz number, then
draws predicted-vs-measured kappa: metals hug the L = 2.44e-8 line, while a phonon insulator
flies off it.

    python examples/wiedemann_franz_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wiedemann_franz import (lorenz_number, thermal_conductivity,  # noqa: E402
                             effective_lorenz, obeys_law)


# (name, sigma S/m, measured kappa W/mK)
METALS = [
    ("silver", 6.30e7, 429.0),
    ("copper", 5.96e7, 401.0),
    ("gold", 4.10e7, 318.0),
    ("aluminium", 3.77e7, 237.0),
    ("iron", 1.00e7, 80.0),
    ("lead", 4.55e6, 35.0),
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Wiedemann-Franz: kappa / (sigma T) = L = %.3e W ohm / K^2 (T = 300 K)\n"
          % lorenz_number())
    print(f"  {'metal':<12}{'sigma (S/m)':>13}{'kappa pred':>12}{'kappa meas':>12}{'L_eff/L':>9}")
    for name, sigma, kmeas in METALS:
        kpred = thermal_conductivity(sigma, 300.0)
        Leff = effective_lorenz(kmeas, sigma, 300.0)
        flag = "" if obeys_law(kmeas, sigma, 300.0) else "  (off)"
        print(f"  {name:<12}{sigma:>13.2e}{kpred:>10.0f}  {kmeas:>10.0f}  {Leff/lorenz_number():>7.2f}{flag}")

    print("\n  The same free electrons carry both currents, so the material-specific mean free")
    print("  path and carrier density cancel: kappa/(sigma T) is a near-universal constant. That")
    print("  lets you read a metal's heat conductivity off an easy resistance measurement -- and")
    print("  a measured L well below 2.44e-8 flags heat and charge decoupling (strange metals).")

    _svg(os.path.join(outdir, "wiedemann_franz.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'wiedemann_franz.svg')}")


def _svg(path, size=720, pad=82):
    # predicted (x) vs measured (y) thermal conductivity; the WF line is y = x.
    kmax = 480.0
    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 44

    def X(k):
        return x0 + k / kmax * (x1 - x0)

    def Y(k):
        return y0 - k / kmax * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Wiedemann-Franz: predicted vs measured</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'metals fall on the L = 2.44e-8 line; heat and charge ride the same electrons</text>',
    ]

    # axes
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.5"/>')
    for k in range(0, 481, 100):
        parts.append(f'<text x="{X(k):.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{k}</text>')
        parts.append(f'<text x="{x0-8:.1f}" y="{Y(k)+4:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="end">{k}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+30:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">predicted kappa = L sigma T (W/m K)</text>')
    parts.append(f'<text x="24" y="{(y0+y1)/2:.1f}" fill="#8b949e" font-size="11" '
                 f'transform="rotate(-90 24 {(y0+y1)/2:.1f})" text-anchor="middle">measured kappa (W/m K)</text>')

    # y = x Wiedemann-Franz line
    parts.append(f'<line x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(kmax):.1f}" y2="{Y(kmax):.1f}" '
                 f'stroke="#ffd43b" stroke-width="1.6" stroke-dasharray="6 4"/>')
    parts.append(f'<text x="{X(360):.1f}" y="{Y(360)-8:.1f}" fill="#ffd43b" font-size="11">'
                 f'WF law (L = 2.44e-8)</text>')

    for name, sigma, kmeas in METALS:
        kpred = thermal_conductivity(sigma, 300.0)
        parts.append(f'<circle cx="{X(kpred):.1f}" cy="{Y(kmeas):.1f}" r="5" fill="#4dabf7"/>')
        parts.append(f'<text x="{X(kpred)+8:.1f}" y="{Y(kmeas)+4:.1f}" fill="#8b949e" '
                     f'font-size="10">{name}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
