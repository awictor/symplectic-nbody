"""Demo: magnetic braking and gyrochronology.

Prints gyrochronology ages for real rotators and clusters, then draws the
Skumanich spin-down curve -- rotation period vs stellar age -- with the Sun and a
few benchmark clusters marked on it.

    python examples/magnetic_braking_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from magnetic_braking import (rotation_period, gyro_age, alfven_radius,  # noqa: E402
                              P_SUN, T_SUN, DAY, GYR)

R_SUN = 6.957e8


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Magnetic braking & gyrochronology: a star's spin is its clock\n")
    print("  Skumanich law: P(t) = P_sun sqrt(t / t_sun), so t = t_sun (P/P_sun)^2\n")
    print(f"  {'rotator':>22}{'P (days)':>10}{'gyro age (Gyr)':>16}")
    print("  " + "-" * 48)
    rotators = [
        ("Pleiades member", 3.0),
        ("young field star", 6.0),
        ("Hyades member", 8.5),
        ("the Sun", 25.4),
        ("old thick-disk star", 35.0),
    ]
    for name, p_days in rotators:
        age = gyro_age(p_days * DAY) / GYR
        print(f"  {name:>22}{p_days:>10.1f}{age:>16.2f}")

    rA = alfven_radius(2e-4, R_SUN, 2e9, 4e5) / R_SUN
    print(f"\n  solar-wind Alfven radius ~ {rA:.0f} R_sun -- the long lever arm that")
    print("  lets a feeble ~1e-14 Msun/yr mass loss brake the whole star. Fast")
    print("  rotators brake hardest, so a broad spread of young spins converges")
    print("  onto one age-period sequence -- which is what makes the clock work.")

    _svg(os.path.join(outdir, "magnetic_braking.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'magnetic_braking.svg')}")


def _svg(path, size=720, pad=64):
    # period (days) vs age (Gyr), from 0.05 to 10 Gyr
    ages = [0.05 * (1.12 ** i) for i in range(0, 46)]  # ~0.05 .. ~10 Gyr
    ages = [a for a in ages if a <= 12.0]
    pers = [rotation_period(a * GYR) / DAY for a in ages]
    la = [math.log10(a) for a in ages]
    lp = [math.log10(p) for p in pers]
    amin, amax = la[0], la[-1]
    pmin, pmax = min(lp), max(lp)

    def sx(x):
        return pad + (x - amin) / (amax - amin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - pmin) / (pmax - pmin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(la[i]):.1f},{sy(lp[i]):.1f}" for i in range(len(ages)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#8338ec" stroke-width="2.4"/>')

    marks = [("Pleiades", 0.12, "#4dabf7"), ("Hyades", 0.65, "#06d6a0"),
             ("Sun", 4.567, "#ffd43b")]
    for label, age, col in marks:
        p = rotation_period(age * GYR) / DAY
        px = sx(math.log10(age))
        py = sy(math.log10(p))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{col}"/>')
        parts.append(f'<text x="{px+8:.1f}" y="{py+4:.1f}" fill="{col}" '
                     f'font-size="11">{label} ({p:.0f} d)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Gyrochronology: rotation period vs stellar age</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'Skumanich P ~ t^(1/2): a slow spin means an old star</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 age (Gyr) -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 rotation period (days)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
