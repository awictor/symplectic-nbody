"""Demo: tidal locking timescales across the solar system.

Prints locking times for several moons and planets, then draws the brutal a^6
locking-time-vs-distance curve with the age of the solar system marked, so the
line between "locked" and "free" bodies is visible.

    python examples/tidal_locking_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tidal_locking import (locking_time, moment_of_inertia,  # noqa: E402
                           max_locking_distance, M_EARTH, M_MOON, R_MOON,
                           R_EARTH, A_MOON, AGE_SOLAR_SYSTEM, GYR)

DAY = 86400.0


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Tidal locking: t_lock ~ a^6 I Q / (G M_p^2 k2 R^5)\n")
    print(f"  age of the solar system: {AGE_SOLAR_SYSTEM/GYR:.2f} Gyr\n")
    print(f"  {'body -> primary':>24}{'t_lock (Gyr)':>14}{'locked?':>10}")
    print("  " + "-" * 48)

    # (name, satellite M, R, initial spin period hr, orbit a, primary M)
    cases = [
        ("Moon -> Earth", M_MOON, R_MOON, 5.0, A_MOON, M_EARTH, 0.4),
        ("Phobos -> Mars", 1.06e16, 1.11e4, 6.0, 9.38e6, 6.417e23, 0.4),
        ("Io -> Jupiter", 8.93e22, 1.822e6, 8.0, 4.217e8, 1.898e27, 0.38),
        ("Earth -> Moon", M_EARTH, R_EARTH, 24.0, A_MOON, M_MOON, 0.33),
        ("Mercury -> Sun", 3.301e23, 2.44e6, 20.0, 5.79e10, 1.989e30, 0.35),
    ]
    for name, M, R, per_hr, a, Mp, f in cases:
        wi = 2 * math.pi / (per_hr * 3600.0)
        I = moment_of_inertia(M, R, f)
        t = locking_time(wi, a, R, I, Mp) / GYR
        locked = "yes" if t <= AGE_SOLAR_SYSTEM / GYR else "no"
        tstr = f"{t:.3g}"
        print(f"  {name:>24}{tstr:>14}{locked:>10}")

    print("\n  The a^6 dependence is everything: close-in moons lock in a geological")
    print("  blink, while the Earth (braking only on the Moon's weak tide) needs far")
    print("  longer than the universe is old. Mercury dodged full locking into a 3:2")
    print("  spin-orbit resonance; hot Jupiters at a few stellar radii are all locked.")

    _svg(os.path.join(outdir, "tidal_locking.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'tidal_locking.svg')}")


def _svg(path, size=720, pad=64):
    # locking time vs distance for a Moon-like body around Earth, log-log
    wi = 2 * math.pi / (5 * 3600.0)
    I = moment_of_inertia(M_MOON, R_MOON)
    a_re = [2.0 * (1.1 ** i) for i in range(0, 60)]   # in R_earth
    a_re = [x for x in a_re if x <= 400.0]
    ts = [locking_time(wi, x * R_EARTH, R_MOON, I, M_EARTH) / GYR for x in a_re]
    lx = [math.log10(x) for x in a_re]
    ly = [math.log10(t) for t in ts]
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
    # age-of-solar-system horizontal line
    age = math.log10(AGE_SOLAR_SYSTEM / GYR)
    if ymin <= age <= ymax:
        ya = sy(age)
        parts.append(f'<line x1="{pad}" y1="{ya:.1f}" x2="{size-pad}" y2="{ya:.1f}" '
                     f'stroke="#ff6b6b" stroke-width="1.3" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{pad+8}" y="{ya-6:.1f}" fill="#ff6b6b" '
                     f'font-size="11">age of the solar system (4.57 Gyr)</text>')

    poly = " ".join(f"{sx(lx[i]):.1f},{sy(ly[i]):.1f}" for i in range(len(a_re)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#8338ec" stroke-width="2.4"/>')

    # mark the Moon's actual distance (60 R_earth)
    mx = sx(math.log10(60.0))
    mt = locking_time(wi, 60.0 * R_EARTH, R_MOON, I, M_EARTH) / GYR
    my = sy(math.log10(mt))
    parts.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="5" fill="#ffd43b"/>')
    parts.append(f'<text x="{mx+8:.1f}" y="{my+4:.1f}" fill="#ffd43b" '
                 f'font-size="11">Moon (60 R_earth)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Tidal locking time ~ a^6</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'below the red line: locked; above it: still spinning freely</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 distance (R_earth) -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 locking time (Gyr)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
