"""Demo: the free-fall / dynamical timescale across the cosmos.

Prints free-fall times for objects spanning 20 orders of magnitude in density, then
draws t_ff vs mean density -- one straight line on a log-log plot, because every
self-gravitating system ticks on the same 1/sqrt(G rho) clock.

    python examples/free_fall_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from free_fall import free_fall_time, mean_density, M_SUN, R_SUN  # noqa: E402

MINUTE = 60.0
YEAR = 3.15576e7
KYR = 1e3 * YEAR
MYR = 1e6 * YEAR
M_P = 1.6726219e-27


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Free-fall time t_ff = sqrt(3 pi / 32 G rho): the clock of gravity\n")
    print(f"  {'system':>24}{'rho (kg/m^3)':>15}{'t_ff':>18}")
    print("  " + "-" * 57)
    systems = [
        ("giant molecular cloud", 1e2 * 1e6 * 2.3 * M_P),
        ("dense cloud core", 1e4 * 1e6 * 2.3 * M_P),
        ("protostellar core", 1e8 * 1e6 * 2.3 * M_P),
        ("the Sun (mean)", mean_density(M_SUN, R_SUN)),
        ("the Earth", mean_density(5.972e24, 6.371e6)),
        ("white dwarf", 1e9),
        ("neutron star", 5e17),
    ]
    for name, rho in systems:
        t = free_fall_time(rho)
        print(f"  {name:>24}{rho:>15.2e}{_fmt(t):>18}")

    print("\n  Notice: t_ff depends only on density, not size or mass. A galaxy and a")
    print("  raindrop of the same mean density collapse in the same time. That is why")
    print("  low Earth orbit is always ~90 minutes, why a molecular cloud core forms")
    print("  stars in a few hundred kyr, and why a neutron star's dynamical time is")
    print("  well under a millisecond -- denser means faster, universally.")

    _svg(os.path.join(outdir, "free_fall.svg"), systems)
    print(f"\n  wrote {os.path.join(outdir, 'free_fall.svg')}")


def _fmt(t):
    if t < 1.0:
        return f"{t*1e3:.2f} ms"
    if t < 2 * MINUTE:
        return f"{t:.1f} s"
    if t < 2 * 3600:
        return f"{t/MINUTE:.1f} min"
    if t < 2 * YEAR:
        return f"{t/86400:.1f} days"
    if t < 2 * MYR:
        return f"{t/KYR:.1f} kyr"
    return f"{t/MYR:.2f} Myr"


def _svg(path, systems, size=720, pad=70):
    rhos = [10 ** (-19 + 0.4 * i) for i in range(0, 96)]  # 1e-19 .. 1e18
    ts = [free_fall_time(r) for r in rhos]
    lx = [math.log10(r) for r in rhos]
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
    poly = " ".join(f"{sx(lx[i]):.1f},{sy(ly[i]):.1f}" for i in range(len(rhos)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#8338ec" stroke-width="2.6"/>')

    labels = [("cloud", systems[0][1], "#4dabf7"),
              ("Sun", systems[3][1], "#ffd43b"),
              ("Earth", systems[4][1], "#06d6a0"),
              ("neutron star", systems[6][1], "#ff6b6b")]
    for name, rho, col in labels:
        t = free_fall_time(rho)
        px = sx(math.log10(rho))
        py = sy(math.log10(t))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{col}"/>')
        parts.append(f'<text x="{px+8:.1f}" y="{py-6:.1f}" fill="{col}" '
                     f'font-size="11">{name}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'The universal clock of gravity: t_ff ~ rho^(-1/2)</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'one line over 37 decades in density -- size and mass drop out</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 mean density (kg/m^3) -&gt;</text>')
    parts.append(f'<text x="{pad-16}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 free-fall time (s)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
