"""Demo: Rayleigh-Benard convection onset and heat transport.

Prints the Rayleigh number and convection state for systems from a thin lab cell to
the solar convection zone, then draws the Nusselt number vs Rayleigh number, flat at
1 (conduction) up to Ra_c ~ 1708 and then climbing as (Ra/Ra_c)^(1/3).

    python examples/rayleigh_benard_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rayleigh_benard import (rayleigh_number, is_convecting,  # noqa: E402
                             critical_delta_T, nusselt_number,
                             RA_CRITICAL_RIGID)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Rayleigh-Benard: Ra = g alpha dT d^3 / (nu kappa); onset at Ra_c ~ 1708\n")
    print(f"  {'system':>22}{'Ra':>12}{'state':>16}{'Nu':>8}")
    print("  " + "-" * 58)
    # (name, dT, d, alpha, nu, kappa)
    systems = [
        ("lab cell (near onset)", 0.002, 0.01, 2.6e-4, 1e-6, 1.4e-7),
        ("mug of coffee", 5.0, 0.08, 2.6e-4, 1e-6, 1.4e-7),
        ("pot on a stove", 20.0, 0.10, 2.6e-4, 1e-6, 1.4e-7),
        ("Earth's mantle", 3000.0, 2.9e6, 3e-5, 1e17, 1e-6),
        ("solar convection zone", 1e5, 2e8, 1e-4, 1e-2, 1e-2),
    ]
    for name, dT, d, alpha, nu, kappa in systems:
        Ra = rayleigh_number(dT, d, alpha, nu, kappa)
        state = "convecting" if is_convecting(Ra) else "conducting"
        Nu = nusselt_number(Ra)
        nustr = f"{Nu:.0f}" if Nu < 1e5 else f"{Nu:.0e}"
        print(f"  {name:>22}{Ra:>12.1e}{state:>16}{nustr:>10}")

    print("\n  Below Ra_c the layer just conducts (Nu = 1); above it convection")
    print("  switches on sharply and carries far more heat (Nu ~ (Ra/Ra_c)^(1/3)).")
    print("  Astrophysical layers -- the mantle, the solar convection zone -- run")
    print("  at Ra of 10^20 or more, so they are violently, turbulently convective:")
    print("  the granulation on the Sun and the plates under our feet both follow.")

    _svg(os.path.join(outdir, "rayleigh_benard.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'rayleigh_benard.svg')}")


def _svg(path, size=720, pad=70):
    Ras = [10 ** (2 + 0.1 * i) for i in range(0, 81)]   # 1e2 .. 1e10
    Nus = [nusselt_number(Ra) for Ra in Ras]
    lx = [math.log10(Ra) for Ra in Ras]
    ly = [math.log10(Nu) for Nu in Nus]
    xmin, xmax = lx[0], lx[-1]
    ymin, ymax = 0.0, max(ly) * 1.05

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
    # onset vertical line at Ra_c
    xc = sx(math.log10(RA_CRITICAL_RIGID))
    parts.append(f'<line x1="{xc:.1f}" y1="{pad}" x2="{xc:.1f}" y2="{size-pad}" '
                 f'stroke="#ff6b6b" stroke-width="1.4" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{xc+6:.1f}" y="{pad+16:.1f}" fill="#ff6b6b" '
                 f'font-size="11">Ra_c ~ 1708 (onset)</text>')

    poly = " ".join(f"{sx(lx[i]):.1f},{sy(ly[i]):.1f}" for i in range(len(Ras)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ffd43b" stroke-width="2.6"/>')

    parts.append(f'<text x="{pad+10}" y="{sy(0.0)-8:.1f}" fill="#8b949e" '
                 f'font-size="10">Nu = 1 (pure conduction)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Heat transport across the convective onset</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'Nu flat at 1 below Ra_c, then climbing as (Ra/Ra_c)^(1/3)</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 Rayleigh number -&gt;</text>')
    parts.append(f'<text x="{pad-14}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 Nusselt number</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
