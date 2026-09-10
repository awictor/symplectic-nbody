"""Demo: the Parker transonic solar wind.

Prints the sound speed, critical radius and wind speed vs distance for a few coronal
temperatures, then draws the transonic velocity profiles passing through Mach 1 at the
critical radius -- the solutions that proved the corona must expand.

    python examples/parker_wind_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parker_wind import (sound_speed, critical_radius, wind_mach,  # noqa: E402
                         wind_speed, R_SUN, AU)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Parker wind: the transonic solution through the sonic critical point\n")
    print(f"  {'T (MK)':>8}{'c_s (km/s)':>12}{'r_c (R_sun)':>13}"
          f"{'v(1 AU) km/s':>15}{'Mach(1 AU)':>12}")
    print("  " + "-" * 60)
    for T_mk in (1.0, 1.5, 2.0, 3.0):
        T = T_mk * 1e6
        cs = sound_speed(T) / 1e3
        rc = critical_radius(T) / R_SUN
        v = wind_speed(AU, T) / 1e3
        M = wind_mach(AU, T)
        print(f"  {T_mk:>8.1f}{cs:>12.1f}{rc:>13.2f}{v:>15.0f}{M:>12.2f}")

    print("\n  A hotter corona has a larger sound speed but a SMALLER critical radius")
    print("  (r_c ~ 1/c_s^2), so the wind goes supersonic sooner and reaches a higher")
    print("  terminal speed. Parker's key point: the only solution that stays finite")
    print("  at both ends passes smoothly through Mach 1 -- a static corona is")
    print("  impossible, and the supersonic solar wind confirmed by Mariner 2 follows.")

    _svg(os.path.join(outdir, "parker_wind.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'parker_wind.svg')}")


def _svg(path, size=720, pad=68):
    temps = [(1.0e6, "#4dabf7", "1 MK"), (1.5e6, "#06d6a0", "1.5 MK"),
             (2.0e6, "#ffd43b", "2 MK"), (3.0e6, "#ff6b6b", "3 MK")]
    # radius axis in R_sun, log from ~1 to 1 AU
    rs = [R_SUN * (1.15 ** i) for i in range(0, 55)]
    rs = [r for r in rs if r <= AU]
    curves = []
    for T, col, label in temps:
        ys = [wind_speed(r, T) / 1e3 for r in rs]
        curves.append((ys, col, label, T))

    lx = [math.log10(r / R_SUN) for r in rs]
    all_y = [y for ys, _, _, _ in curves for y in ys]
    xmin, xmax = lx[0], lx[-1]
    ymin, ymax = 0.0, max(all_y) * 1.05

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
    ytop = pad + 22
    for i, (ys, col, label, T) in enumerate(curves):
        poly = " ".join(f"{sx(lx[j]):.1f},{sy(ys[j]):.1f}" for j in range(len(rs)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2.3"/>')
        # mark the critical point (Mach 1) on each curve
        rc = critical_radius(T)
        if R_SUN <= rc <= AU:
            cx = sx(math.log10(rc / R_SUN))
            cy = sy(sound_speed(T) / 1e3)
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{col}" '
                         f'stroke="#0d1117" stroke-width="1"/>')
        parts.append(f'<text x="{pad+10}" y="{ytop + i*16}" fill="{col}" '
                     f'font-size="12">{label}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Parker wind: transonic velocity profiles</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'dots mark the sonic critical point (Mach 1); hotter = faster wind</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 radius (R_sun) -&gt;</text>')
    parts.append(f'<text x="{pad-14}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'wind speed (km/s)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
