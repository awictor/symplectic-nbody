"""Demo: the gravity-assist slingshot.

Prints the turn angle and heliocentric boost for Jupiter flybys of varying depth and
speed, then draws the maximum speed change vs periapsis distance -- deeper, slower
passes bend more and steal more of the planet's orbital motion.

    python examples/gravity_assist_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gravity_assist import (eccentricity, turn_angle, max_delta_v,  # noqa: E402
                            heliocentric_speed_after, MU_JUPITER, V_JUPITER)

R_JUP = 7.1492e7


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Gravity assist: bend v_inf in the planet's frame, gain in the Sun's\n")
    print(f"  Jupiter orbital speed {V_JUPITER/1e3:.1f} km/s; max possible gain = 2 v_inf\n")
    print(f"  {'v_inf (km/s)':>13}{'periapsis':>12}{'turn (deg)':>12}"
          f"{'boost (km/s)':>14}")
    print("  " + "-" * 51)
    for v_inf in (5e3, 10e3, 15e3):
        for rp_mult in (2.0, 5.0):
            rp = rp_mult * R_JUP
            delta = math.degrees(turn_angle(v_inf, rp, MU_JUPITER))
            dv = max_delta_v(v_inf, rp, MU_JUPITER) / 1e3
            print(f"  {v_inf/1e3:>13.0f}{f'{rp_mult:.0f} R_jup':>12}"
                  f"{delta:>12.1f}{dv:>14.2f}")

    print("\n  A slower, deeper pass swings the velocity vector through a bigger angle")
    print("  and adds more of Jupiter's 13 km/s orbital motion -- up to 2 v_inf for a")
    print("  near-reversal. Voyager 2 chained Jupiter, Saturn, Uranus and Neptune this")
    print("  way to reach Solar-System escape speed on a fraction of the fuel a direct")
    print("  burn would need, while the planets lost a laughably tiny bit of energy.")

    _svg(os.path.join(outdir, "gravity_assist.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'gravity_assist.svg')}")


def _svg(path, size=720, pad=72):
    rps = [1.5 + 0.3 * i for i in range(0, 60)]   # periapsis in R_jup, 1.5 .. ~19
    curves = [(5e3, "#4dabf7", "v_inf = 5 km/s"),
              (10e3, "#ffd43b", "v_inf = 10 km/s"),
              (15e3, "#ff6b6b", "v_inf = 15 km/s")]
    data = []
    for v_inf, col, label in curves:
        ys = [max_delta_v(v_inf, rp * R_JUP, MU_JUPITER) / 1e3 for rp in rps]
        data.append((ys, col, label))

    xmin, xmax = rps[0], rps[-1]
    ymax = max(y for ys, _, _ in data for y in ys) * 1.05
    ymin = 0.0

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
    # 2 v_inf ceilings (dashed) for each curve
    ytop = pad + 22
    for i, (ys, col, label) in enumerate(data):
        poly = " ".join(f"{sx(rps[j]):.1f},{sy(ys[j]):.1f}" for j in range(len(rps)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        parts.append(f'<text x="{size-pad-140}" y="{ytop + i*16}" fill="{col}" '
                     f'font-size="12">{label}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Slingshot boost vs flyby periapsis</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'deeper, slower passes bend more and steal more of the planet\'s motion</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">periapsis distance (Jupiter radii) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'max heliocentric boost (km/s)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
