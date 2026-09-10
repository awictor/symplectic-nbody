"""Demo: the constant-1-g relativistic rocket.

Prints ship (proper) time, Earth time and peak velocity to reach destinations from
Proxima to Andromeda at a constant 1 g, then draws ship time vs distance beside Earth
time -- the two diverge dramatically as relativity lets the crew outrun the clock.

    python examples/relativistic_rocket_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from relativistic_rocket import (velocity, lorentz_gamma, earth_time,  # noqa: E402
                                 proper_time_for_distance, C, G_EARTH, YEAR, LY)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    g = G_EARTH
    print("Relativistic rocket at constant 1 g (acceleration only)\n")
    print(f"  {'destination':>20}{'distance':>12}{'ship (yr)':>11}"
          f"{'Earth (yr)':>13}{'v_peak/c':>10}")
    print("  " + "-" * 66)
    dests = [
        ("Proxima Centauri", 4.37),
        ("Vega", 25.0),
        ("Galactic centre", 27000.0),
        ("Andromeda (M31)", 2.5e6),
        ("edge of observable", 4.6e10),
    ]
    for name, ly in dests:
        d = ly * LY
        tau = proper_time_for_distance(g, d)
        t_e = earth_time(g, tau)
        v = velocity(g, tau) / C
        dstr = f"{ly:.2f} ly" if ly < 1e3 else (f"{ly/1e3:.0f} kly" if ly < 1e6 else f"{ly/1e6:.2g} Mly")
        print(f"  {name:>20}{dstr:>12}{tau/YEAR:>11.1f}{t_e/YEAR:>13.3g}{v:>10.4f}")

    print("\n  Velocity is c tanh(a tau/c) -- it saturates just short of c -- but proper")
    print("  time uses cosh/sinh, so the crew's clock falls ever further behind Earth's.")
    print("  A 1-g ship reaches the galactic centre in ~10 crew-years (27,000 pass on")
    print("  Earth) and could cross to Andromeda in ~15. The impossible part is fuel: a")
    print("  photon drive needs exp(2 phi) times the payload mass, astronomically much.")

    _svg(os.path.join(outdir, "relativistic_rocket.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'relativistic_rocket.svg')}")


def _svg(path, size=720, pad=72):
    g = G_EARTH
    lys = [10 ** (0.0 + 0.12 * i) for i in range(0, 92)]   # 1 ly .. ~1e11 ly
    ship = [proper_time_for_distance(g, ly * LY) / YEAR for ly in lys]
    earth = [earth_time(g, proper_time_for_distance(g, ly * LY)) / YEAR for ly in lys]
    lx = [math.log10(ly) for ly in lys]
    ls = [math.log10(t) for t in ship]
    le = [math.log10(t) for t in earth]
    xmin, xmax = lx[0], lx[-1]
    ymin, ymax = min(ls), max(le)

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
    ps = " ".join(f"{sx(lx[i]):.1f},{sy(ls[i]):.1f}" for i in range(len(lys)))
    pe = " ".join(f"{sx(lx[i]):.1f},{sy(le[i]):.1f}" for i in range(len(lys)))
    parts.append(f'<polyline points="{pe}" fill="none" stroke="#ff6b6b" stroke-width="2.4"/>')
    parts.append(f'<polyline points="{ps}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')

    parts.append(f'<text x="{pad+10}" y="{pad+22}" fill="#ff6b6b" font-size="12">'
                 f'Earth time (~ distance in ly)</text>')
    parts.append(f'<text x="{pad+10}" y="{pad+38}" fill="#4dabf7" font-size="12">'
                 f'ship (proper) time -- logarithmic!</text>')

    for name, ly, col in [("gal. centre", 2.7e4, "#ffd43b"), ("Andromeda", 2.5e6, "#06d6a0")]:
        tau = proper_time_for_distance(g, ly * LY) / YEAR
        px = sx(math.log10(ly))
        py = sy(math.log10(tau))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{col}"/>')
        parts.append(f'<text x="{px+8:.1f}" y="{py+4:.1f}" fill="{col}" '
                     f'font-size="10">{name} ({tau:.0f} yr)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Constant-1-g travel: ship time vs Earth time</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'ship time grows only logarithmically with distance -- the Galaxy in a lifetime</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 distance (light-years) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 elapsed time (years)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
