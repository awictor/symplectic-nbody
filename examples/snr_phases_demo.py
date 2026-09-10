"""Demo: the four phases of a supernova remnant.

Prints the radius, speed and phase at ages from centuries to a million years, then
draws the radius-vs-time track with the free-expansion, Sedov and snowplow slopes and
the phase-transition boundaries marked.

    python examples/snr_phases_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from snr_phases import (ism_density, sweep_up_radius,  # noqa: E402
                        free_expansion_end_time, sedov_radius, sedov_velocity,
                        phase, merge_radius, PC, M_SUN, YEAR, E_SN)

M_EJ = 5 * M_SUN
V_EJ = 1e7  # 10,000 km/s


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    rho = ism_density(1.0)
    print("Supernova remnant: free expansion -> Sedov -> snowplow -> merge\n")
    print(f"  E = 1e51 erg, M_ej = 5 Msun, n = 1 /cc\n")
    print(f"  sweep-up radius (free expansion ends): "
          f"{sweep_up_radius(M_EJ, rho)/PC:.1f} pc at "
          f"{free_expansion_end_time(M_EJ, rho, V_EJ)/YEAR:.0f} yr")
    print(f"  merge radius (~10 km/s): {merge_radius()/PC:.0f} pc\n")
    print(f"  {'age (yr)':>12}{'R (pc)':>10}{'v (km/s)':>12}{'phase':>16}")
    print("  " + "-" * 50)
    for t_yr in (100, 300, 1000, 5000, 2e4, 5e4, 2e5, 1e6):
        t = t_yr * YEAR
        if t < free_expansion_end_time(M_EJ, rho, V_EJ):
            R, v = V_EJ * t, V_EJ
        else:
            R, v = sedov_radius(t), sedov_velocity(t)
        ph = phase(t, M_EJ, V_EJ)
        print(f"  {t_yr:>12.0f}{R/PC:>10.1f}{v/1e3:>12.0f}{ph:>16}")

    print("\n  The blast coasts ballistically for a few centuries, then sweeps up")
    print("  enough gas to enter the long adiabatic Sedov phase (R ~ t^2/5) for tens")
    print("  of thousands of years. When the shell cools it radiates and coasts on")
    print("  momentum (snowplow), finally merging into the ISM near 100 pc after a")
    print("  million years -- seeding the galaxy with the elements it forged.")

    _svg(os.path.join(outdir, "snr_phases.svg"), rho)
    print(f"\n  wrote {os.path.join(outdir, 'snr_phases.svg')}")


def _svg(path, rho, size=720, pad=70):
    t_free = free_expansion_end_time(M_EJ, rho, V_EJ)
    ts = [10 ** (1.5 + 0.08 * i) * YEAR for i in range(0, 76)]  # ~30 yr .. ~1e6.5 yr
    Rs = []
    for t in ts:
        if t < t_free:
            Rs.append(V_EJ * t)
        else:
            Rs.append(sedov_radius(t))
    lx = [math.log10(t / YEAR) for t in ts]
    ly = [math.log10(R / PC) for R in Rs]
    xmin, xmax = lx[0], lx[-1]
    ymin, ymax = min(ly), max(ly)

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
    # phase boundaries
    for t_yr, label, col in [(t_free / YEAR, "free -> Sedov", "#4dabf7"),
                             (3e4, "Sedov -> snowplow", "#ffd43b"),
                             (1e6, "merge", "#ff6b6b")]:
        if xmin <= math.log10(t_yr) <= xmax:
            bx = sx(math.log10(t_yr))
            parts.append(f'<line x1="{bx:.1f}" y1="{pad}" x2="{bx:.1f}" y2="{size-pad}" '
                         f'stroke="{col}" stroke-width="1.2" stroke-dasharray="4 4"/>')
            parts.append(f'<text x="{bx+4:.1f}" y="{pad+16:.1f}" fill="{col}" '
                         f'font-size="10" transform="rotate(90 {bx+4:.1f} {pad+16:.1f})">'
                         f'{label}</text>')

    poly = " ".join(f"{sx(lx[i]):.1f},{sy(ly[i]):.1f}" for i in range(len(ts)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#8338ec" stroke-width="2.6"/>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Supernova remnant: radius vs age</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'ballistic R~t, then Sedov R~t^(2/5), then snowplow R~t^(2/7)</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 age (yr) -&gt;</text>')
    parts.append(f'<text x="{pad-16}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 radius (pc)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
