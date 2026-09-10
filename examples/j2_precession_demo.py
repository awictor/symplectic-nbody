"""Demo: J2 orbital precession from a planet's oblateness.

Prints the nodal and apsidal precession rates for benchmark orbits, then draws both
rates vs inclination, marking the sun-synchronous inclination (nodal drift = Sun's
motion) and the 63.4-degree critical inclination (frozen apsides, the Molniya orbit).

    python examples/j2_precession_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from j2_precession import (nodal_precession_rate, apsidal_precession_rate,  # noqa: E402
                           sun_synchronous_inclination, critical_inclination,
                           DEG_PER_DAY, SUN_RATE_DEG_DAY, R_EARTH)

A_LEO = (6378 + 700) * 1e3


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("J2 orbital precession (Earth, 700 km circular orbit)\n")
    print(f"  {'orbit':>22}{'incl (deg)':>12}{'nodal (deg/d)':>15}"
          f"{'apsidal (deg/d)':>17}")
    print("  " + "-" * 66)
    cases = [
        ("equatorial", 0.0),
        ("ISS-like", 51.6),
        ("critical (Molniya)", math.degrees(critical_inclination())),
        ("sun-synchronous", math.degrees(sun_synchronous_inclination(A_LEO))),
        ("polar", 90.0),
    ]
    for name, i_deg in cases:
        i = math.radians(i_deg)
        nod = nodal_precession_rate(A_LEO, 0.0, i) * DEG_PER_DAY
        aps = apsidal_precession_rate(A_LEO, 0.0, i) * DEG_PER_DAY
        print(f"  {name:>22}{i_deg:>12.2f}{nod:>15.3f}{aps:>17.3f}")

    print(f"\n  The equatorial bulge (J2) drags orbit planes and rotates ellipses. Tune")
    print("  the inclination so the nodal drift equals the Sun's 0.9856 deg/day and the")
    print("  orbit stays fixed relative to the Sun -- a sun-synchronous orbit crossing")
    print("  the equator at the same local time daily. At 63.4 deg the apsides freeze")
    print("  (Molniya), parking apogee over the far north for long dwell times.")

    _svg(os.path.join(outdir, "j2_precession.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'j2_precession.svg')}")


def _svg(path, size=720, pad=72):
    incs = [i for i in range(0, 181)]
    nod = [nodal_precession_rate(A_LEO, 0.0, math.radians(i)) * DEG_PER_DAY for i in incs]
    aps = [apsidal_precession_rate(A_LEO, 0.0, math.radians(i)) * DEG_PER_DAY for i in incs]
    xmin, xmax = 0.0, 180.0
    allv = nod + aps
    ymin, ymax = min(allv), max(allv)
    pad_y = 0.1 * (ymax - ymin)
    ymin -= pad_y
    ymax += pad_y

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # zero line
    y0 = sy(0.0)
    parts.append(f'<line x1="{pad}" y1="{y0:.1f}" x2="{size-pad}" y2="{y0:.1f}" stroke="#30363d"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>')

    # sun-sync + critical vertical markers
    i_ss = math.degrees(sun_synchronous_inclination(A_LEO))
    i_c = math.degrees(critical_inclination())
    for i_deg, label, col in [(i_ss, "sun-sync", "#ffd43b"), (i_c, "critical 63.4", "#06d6a0")]:
        x = sx(i_deg)
        parts.append(f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{size-pad}" '
                     f'stroke="{col}" stroke-width="1" stroke-dasharray="4 4"/>')
        parts.append(f'<text x="{x+4:.1f}" y="{pad+40:.1f}" fill="{col}" '
                     f'font-size="10" transform="rotate(90 {x+4:.1f} {pad+40:.1f})">{label}</text>')

    pn = " ".join(f"{sx(incs[i]):.1f},{sy(nod[i]):.1f}" for i in range(len(incs)))
    pa = " ".join(f"{sx(incs[i]):.1f},{sy(aps[i]):.1f}" for i in range(len(incs)))
    parts.append(f'<polyline points="{pn}" fill="none" stroke="#4dabf7" stroke-width="2.4"/>')
    parts.append(f'<polyline points="{pa}" fill="none" stroke="#ff6b6b" stroke-width="2.4"/>')

    parts.append(f'<text x="{pad+10}" y="{pad+22}" fill="#4dabf7" font-size="12">nodal dOmega/dt</text>')
    parts.append(f'<text x="{pad+10}" y="{pad+38}" fill="#ff6b6b" font-size="12">apsidal domega/dt</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'J2 precession rates vs inclination</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">inclination (deg) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'precession rate (deg/day)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
