"""Demo: equipartition and the heat-capacity staircase of a diatomic gas.

Prints molar heat capacities and gamma for monatomic, diatomic and polyatomic gases and a
solid (Dulong-Petit), then draws the H2 heat-capacity staircase: C_V/R climbing from 3/2
(translation) to 5/2 (rotation switches on) toward 7/2 (vibration) as quantum modes thaw.

    python examples/equipartition_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from equipartition import (molar_cv, molar_cp, gamma_ratio,  # noqa: E402
                           effective_cv_diatomic, R_GAS)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Equipartition: (1/2) k_B T per quadratic degree of freedom\n")
    print(f"  {'gas / solid':<26}{'f':>3}{'C_V/R':>8}{'C_P/R':>8}{'gamma':>8}")
    rows = [
        ("monatomic (He, Ar)", 3),
        ("diatomic, room T (N2)", 5),
        ("diatomic, hot (+vib)", 7),
        ("nonlinear triatomic", 6),
        ("solid (Dulong-Petit)", 6),
    ]
    for label, f in rows:
        print(f"  {label:<26}{f:>3}{molar_cv(f)/R_GAS:>8.2f}"
              f"{molar_cp(f)/R_GAS:>8.2f}{gamma_ratio(f):>8.3f}")

    print("\n  H2 heat-capacity staircase (theta_rot=85 K, theta_vib=6000 K):")
    print(f"  {'T (K)':>10}{'C_V/R':>10}")
    for T in (20, 50, 100, 300, 1000, 3000, 6000, 10000):
        print(f"  {T:>10}{effective_cv_diatomic(T, 85.0, 6000.0)/R_GAS:>10.2f}")

    print("\n  Classical equipartition is the high-T ceiling; each mode contributes only")
    print("  once k_B T tops its quantum. So H2 sits at 3R/2 while rotation is frozen, steps")
    print("  up to 5R/2 once it thaws (~100 K), and only near ~1000s K does vibration lift it")
    print("  toward 7R/2 -- the staircase Maxwell could see but not explain before quanta.")

    _svg(os.path.join(outdir, "equipartition.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'equipartition.svg')}")


def _svg(path, size=720, pad=78):
    theta_rot, theta_vib = 85.0, 6000.0
    # log-T axis from 10 K to 3e4 K
    lt0, lt1 = 1.0, math.log10(3.0e4)
    n = 240
    Ts = [10 ** (lt0 + (lt1 - lt0) * i / (n - 1)) for i in range(n)]
    cvs = [effective_cv_diatomic(T, theta_rot, theta_vib) / R_GAS for T in Ts]

    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 34
    cv_lo, cv_hi = 1.2, 3.7

    def X(T):
        return x0 + (math.log10(T) - lt0) / (lt1 - lt0) * (x1 - x0)

    def Y(cv):
        return y0 - (cv - cv_lo) / (cv_hi - cv_lo) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The heat-capacity staircase (H2)</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'C_V/R climbs as rotation then vibration thaw -- equipartition is only the hot ceiling</text>',
    ]

    # axes
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.5"/>')

    # plateau gridlines at 3/2, 5/2, 7/2
    for cv, lbl, col in ((1.5, "3R/2  translation", "#4dabf7"),
                         (2.5, "5R/2  + rotation", "#06d6a0"),
                         (3.5, "7R/2  + vibration", "#ff922b")):
        gy = Y(cv)
        parts.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" '
                     f'stroke="{col}" stroke-width="1" stroke-dasharray="5 5" opacity="0.55"/>')
        parts.append(f'<text x="{x1-4:.1f}" y="{gy-5:.1f}" fill="{col}" font-size="11" '
                     f'text-anchor="end">{lbl}</text>')
        parts.append(f'<text x="{x0-8:.1f}" y="{gy+4:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="end">{cv:.1f}</text>')

    # x decade ticks
    for e in range(1, 5):
        T = 10.0 ** e
        gx = X(T)
        parts.append(f'<line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y0+4}" stroke="#8b949e"/>')
        parts.append(f'<text x="{gx:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">10^{e} K</text>')
    # theta markers
    for theta, lbl in ((theta_rot, "theta_rot"), (theta_vib, "theta_vib")):
        gx = X(theta)
        parts.append(f'<line x1="{gx:.1f}" y1="{y0:.1f}" x2="{gx:.1f}" y2="{y1:.1f}" '
                     f'stroke="#8b949e" stroke-width="0.8" stroke-dasharray="2 4" opacity="0.5"/>')
        parts.append(f'<text x="{gx+4:.1f}" y="{y1+12:.1f}" fill="#8b949e" font-size="10">{lbl}</text>')

    # the staircase curve
    poly = " ".join(f"{X(Ts[i]):.1f},{Y(cvs[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ffd43b" stroke-width="2.8"/>')

    parts.append(f'<text x="{x0+6:.1f}" y="{y1-6:.1f}" fill="#8b949e" font-size="11">'
                 f'molar C_V / R  vs  temperature</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
