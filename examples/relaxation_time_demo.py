"""Demo: two-body relaxation across gravitating systems.

Prints crossing, relaxation and evaporation times for systems from binaries to
galaxies, then draws relaxation time vs N with the Hubble-time line, splitting the
collisional systems (globulars, open clusters) from the collisionless ones (galaxies).

    python examples/relaxation_time_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from relaxation_time import (crossing_time, relaxation_time,  # noqa: E402
                             is_collisionless, evaporation_time, PC, MYR, GYR,
                             HUBBLE_TIME)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Two-body relaxation: t_relax ~ (N / 8 ln N) t_cross\n")
    print(f"  Hubble time = {HUBBLE_TIME/GYR:.1f} Gyr (the collisionless threshold)\n")
    print(f"  {'system':>20}{'N':>10}{'t_cross':>11}{'t_relax':>12}{'state':>16}")
    print("  " + "-" * 69)
    systems = [
        ("open cluster", 1000, 2 * PC, 1e3),
        ("globular cluster", int(1e5), 10 * PC, 10e3),
        ("nuclear star cluster", int(1e7), 5 * PC, 100e3),
        ("dwarf galaxy", int(1e8), 1000 * PC, 30e3),
        ("Milky Way", int(1e11), 15000 * PC, 200e3),
    ]
    for name, N, R, v in systems:
        tc = crossing_time(R, v)
        tr = relaxation_time(N, tc)
        state = "collisionless" if is_collisionless(N, tc) else "collisional"
        print(f"  {name:>20}{N:>10.0e}{_fmt(tc):>11}{_fmt(tr):>12}{state:>16}")

    print("\n  t_relax grows almost linearly with N, so small systems relax and")
    print("  mass-segregate fast while big ones never do. A globular cluster relaxes")
    print("  in ~1 Gyr and slowly evaporates; the Milky Way's relaxation time is")
    print("  millions of Hubble times, so it is collisionless -- which is why galaxies")
    print("  keep their spiral arms, streams and cold disks intact for a Hubble time.")

    _svg(os.path.join(outdir, "relaxation_time.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'relaxation_time.svg')}")


def _fmt(t):
    if t < 2 * MYR:
        return f"{t/MYR*1000:.0f} kyr"
    if t < 2000 * MYR:
        return f"{t/MYR:.0f} Myr"
    if t < 1e6 * GYR:
        return f"{t/GYR:.0f} Gyr"
    return f"{t/HUBBLE_TIME:.0e} t_H"


def _svg(path, size=720, pad=70):
    Ns = [10 ** (2 + 0.15 * i) for i in range(0, 67)]   # 1e2 .. 1e12
    # fix a representative t_cross ~ 1 Myr so the N-dependence is clean
    tc = 1.0 * MYR
    trs = [relaxation_time(int(N), tc) for N in Ns]
    lx = [math.log10(N) for N in Ns]
    ly = [math.log10(t / GYR) for t in trs]   # in Gyr, log
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
    ]
    # Hubble-time horizontal line
    lh = math.log10(HUBBLE_TIME / GYR)
    if ymin <= lh <= ymax:
        yh = sy(lh)
        parts.append(f'<rect x="{pad}" y="{pad}" width="{size-2*pad}" '
                     f'height="{yh-pad:.1f}" fill="#4dabf7" fill-opacity="0.08"/>')
        parts.append(f'<line x1="{pad}" y1="{yh:.1f}" x2="{size-pad}" y2="{yh:.1f}" '
                     f'stroke="#ff6b6b" stroke-width="1.4" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{pad+8}" y="{yh-6:.1f}" fill="#ff6b6b" '
                     f'font-size="11">Hubble time -- above: collisionless</text>')

    parts.append(f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>')

    poly = " ".join(f"{sx(lx[i]):.1f},{sy(ly[i]):.1f}" for i in range(len(Ns)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ffd43b" stroke-width="2.6"/>')

    for label, N, col in [("globular", 1e5, "#06d6a0"), ("galaxy", 1e11, "#b197fc")]:
        tr = relaxation_time(int(N), tc)
        px = sx(math.log10(N))
        py = sy(math.log10(tr / GYR))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{col}"/>')
        parts.append(f'<text x="{px+8:.1f}" y="{py+4:.1f}" fill="{col}" '
                     f'font-size="11">{label}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Relaxation time vs particle number (t_cross = 1 Myr)</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f't_relax ~ N/ln N: clusters relax, galaxies never do</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 number of stars N -&gt;</text>')
    parts.append(f'<text x="{pad-16}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 relaxation time (Gyr)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
