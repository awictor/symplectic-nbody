"""Demo: synodic periods and planetary alignments.

Prints the synodic period and conjunction cadence of each planet as seen from Earth,
then draws the synodic period vs sidereal period, showing the divergence at Earth's own
orbit (where alignments never repeat) and the ~1-year floor for distant planets.

    python examples/synodic_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synodic import (synodic_from_earth, conjunctions_per_year,  # noqa: E402
                     synodic_month, synodic_period, P_MERCURY, P_VENUS, P_EARTH,
                     P_MARS, P_JUPITER)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Synodic period: 1/S = |1/P_planet - 1/P_earth| -- when alignments repeat\n")
    print(f"  {'planet':>10}{'sidereal (d)':>14}{'synodic (d)':>14}"
          f"{'per year':>10}")
    print("  " + "-" * 48)
    planets = [("Mercury", P_MERCURY), ("Venus", P_VENUS), ("Mars", P_MARS),
               ("Jupiter", P_JUPITER), ("Saturn", 10759.2), ("Neptune", 60190.0)]
    for name, P in planets:
        S = synodic_from_earth(P)
        print(f"  {name:>10}{P:>14.1f}{S:>14.1f}"
              f"{conjunctions_per_year(P, P_EARTH):>10.3f}")

    print(f"\n  synodic month (new Moon to new Moon): {synodic_month():.2f} days\n")
    print("  Mars returns to opposition every ~780 days -- exactly the ~26-month")
    print("  cadence of Mars launch windows. Inner planets lap Earth quickly; distant")
    print("  planets barely move, so their synodic period approaches one Earth year")
    print("  (Earth does the lapping). Near Earth's own orbit the synodic period blows")
    print("  up: two bodies at the same distance never change their alignment.")

    _svg(os.path.join(outdir, "synodic.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'synodic.svg')}")


def _svg(path, size=720, pad=72):
    # sidereal period from 30 d to 1e5 d (log), synodic period (log)
    Ps = [10 ** (1.5 + 0.05 * i) for i in range(0, 71)]   # ~30 .. ~1e5 d
    # skip a tiny band right at Earth to avoid the singular spike
    Ss = []
    xs = []
    for P in Ps:
        if abs(P - P_EARTH) < 5.0:
            continue
        Ss.append(synodic_period(P, P_EARTH))
        xs.append(P)
    lx = [math.log10(P) for P in xs]
    ly = [math.log10(S) for S in Ss]
    xmin, xmax = math.log10(Ps[0]), math.log10(Ps[-1])
    ymin, ymax = min(ly), min(max(ly), 4.5)

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
    # Earth's period vertical asymptote
    xe = sx(math.log10(P_EARTH))
    parts.append(f'<line x1="{xe:.1f}" y1="{pad}" x2="{xe:.1f}" y2="{size-pad}" '
                 f'stroke="#ff6b6b" stroke-width="1.2" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{xe+5:.1f}" y="{pad+40:.1f}" fill="#ff6b6b" '
                 f'font-size="11">Earth: S -> infinity</text>')
    # 1-year floor
    y1 = sy(math.log10(P_EARTH))
    parts.append(f'<line x1="{pad}" y1="{y1:.1f}" x2="{size-pad}" y2="{y1:.1f}" '
                 f'stroke="#8b949e" stroke-width="0.8" stroke-dasharray="2 4"/>')
    parts.append(f'<text x="{size-pad-4:.1f}" y="{y1-5:.1f}" fill="#8b949e" '
                 f'font-size="10" text-anchor="end">1 year floor</text>')

    # split polyline at Earth (two branches)
    branch, prev_x = [], None
    for i in range(len(xs)):
        if prev_x is not None and (prev_x < P_EARTH) != (xs[i] < P_EARTH):
            parts.append(f'<polyline points="{" ".join(branch)}" fill="none" '
                         f'stroke="#4dabf7" stroke-width="2.4"/>')
            branch = []
        branch.append(f"{sx(lx[i]):.1f},{sy(min(ly[i], ymax)):.1f}")
        prev_x = xs[i]
    if branch:
        parts.append(f'<polyline points="{" ".join(branch)}" fill="none" '
                     f'stroke="#4dabf7" stroke-width="2.4"/>')

    for name, P, col in [("Mercury", P_MERCURY, "#ffd43b"), ("Mars", P_MARS, "#ff922b"),
                         ("Jupiter", P_JUPITER, "#06d6a0")]:
        px = sx(math.log10(P))
        py = sy(math.log10(synodic_period(P, P_EARTH)))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{col}"/>')
        parts.append(f'<text x="{px+8:.1f}" y="{py+4:.1f}" fill="{col}" '
                     f'font-size="11">{name}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Synodic period vs sidereal period (from Earth)</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'diverges at Earth\'s orbit; approaches 1 year for distant planets</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 sidereal period (days) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 synodic period (days)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
