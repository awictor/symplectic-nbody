"""Demo: Alfven waves and the magnetized solar plasma.

Prints Alfven speeds and plasma beta across environments (corona, wind, ISM),
then draws the Alfven speed and bulk-flow speed vs heliocentric distance so the
Alfven surface -- where the wind goes super-Alfvenic and the Sun loses its
magnetic grip -- is visible as the crossing point.

    python examples/alfven_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alfven import (alfven_speed_number_density, plasma_beta,  # noqa: E402
                    alfven_mach, M_P)

R_SUN = 6.957e8
AU = 1.495978707e11


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Alfven waves in magnetized plasma: v_A = B / sqrt(mu0 rho)\n")
    print(f"  {'environment':>22}{'B (T)':>10}{'n (/m^3)':>12}"
          f"{'v_A (km/s)':>12}{'beta':>10}")
    print("  " + "-" * 66)
    envs = [
        ("active corona", 1e-2, 1e15, 2e6),
        ("quiet corona", 1e-3, 1e14, 1e6),
        ("solar wind 1 AU", 5e-9, 5e6, 1e5),
        ("warm ISM", 5e-10, 1e6, 8e3),
    ]
    for name, B, n, T in envs:
        vA = alfven_speed_number_density(B, n)
        beta = plasma_beta(n, T, B)
        print(f"  {name:>22}{B:>10.1e}{n:>12.1e}{vA/1e3:>12.1f}{beta:>10.4f}")

    print("\n  beta < 1 (corona): magnetic tension rules -- the field channels the")
    print("  plasma and stores the energy that heats the corona and drives flares.")
    print("  beta > 1 (dense interiors): gas pressure drags the field around.")
    print("  The solar wind starts sub-Alfvenic (Sun's field co-rotates the plasma")
    print("  and brakes the spin), then crosses the Alfven surface near ~10-20 R_sun")
    print("  and coasts out super-Alfvenic -- decoupled from the Sun's rotation")
    print("  (Parker Solar Probe crossed the real surface near ~15-20 R_sun).")

    _svg(os.path.join(outdir, "alfven.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'alfven.svg')}")


def _wind_profile(r):
    """Toy but monotone radial model: B ~ 1/r^2 (radial field), n ~ 1/r^2
    (mass conservation at ~const speed), u rising from 0 to ~400 km/s.
    Returns (v_A, u) in m/s at heliocentric distance r (m)."""
    x = r / R_SUN
    B = 1e-3 * (1.5 / x) ** 2          # ~10 G at 1.5 R_sun, radial falloff
    n = 3e14 * (1.5 / x) ** 2          # coronal base density, radial falloff
    vA = alfven_speed_number_density(B, n)
    u = 4e5 * (1.0 - math.exp(-(x - 1.0) / 30.0))  # accelerates to 400 km/s
    u = max(u, 1e3)
    return vA, u


def _svg(path, size=720, pad=64):
    xs = [1.5 * (1 + 0.5 * i) for i in range(0, 80)]   # in R_sun, 1.5 .. ~60
    vA = []
    u = []
    for x in xs:
        a, b = _wind_profile(x * R_SUN)
        vA.append(a / 1e3)
        u.append(b / 1e3)
    lx = [math.log10(x) for x in xs]
    allv = vA + u
    lo, hi = min(allv), max(allv)
    xmin, xmax = lx[0], lx[-1]

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - lo) / (hi - lo) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    pa = " ".join(f"{sx(lx[i]):.1f},{sy(vA[i]):.1f}" for i in range(len(xs)))
    pu = " ".join(f"{sx(lx[i]):.1f},{sy(u[i]):.1f}" for i in range(len(xs)))
    parts.append(f'<polyline points="{pa}" fill="none" stroke="#8338ec" stroke-width="2.2"/>')
    parts.append(f'<polyline points="{pu}" fill="none" stroke="#ffd43b" stroke-width="2.2"/>')

    # mark the Alfven surface: first crossing where u exceeds v_A
    for i in range(1, len(xs)):
        if u[i] >= vA[i] and u[i - 1] < vA[i - 1]:
            cx = sx(lx[i])
            parts.append(f'<line x1="{cx:.1f}" y1="{pad}" x2="{cx:.1f}" '
                         f'y2="{size-pad}" stroke="#ff6b6b" stroke-width="1.3" '
                         f'stroke-dasharray="4 4"/>')
            parts.append(f'<text x="{cx+6:.1f}" y="{pad+16:.1f}" fill="#ff6b6b" '
                         f'font-size="11">Alfven surface ({xs[i]:.0f} R_sun)</text>')
            break

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'The Alfven surface: where the solar wind breaks free</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8338ec" font-size="12">'
                 f'v_A (Alfven speed)</text>')
    parts.append(f'<text x="{pad+180}" y="52" fill="#ffd43b" font-size="12">'
                 f'u (wind speed)</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 distance (R_sun) -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'speed (km/s)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
