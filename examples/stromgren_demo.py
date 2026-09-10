"""Demo: the Stromgren sphere around a hot star.

Prints Stromgren radii and ionized masses across densities and stellar types, then
draws R_s vs density for three stars, showing the R ~ n^(-2/3) fall-off from diffuse
diffuse nebulae down to compact HII regions.

    python examples/stromgren_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stromgren import (stromgren_radius, stromgren_radius_pc,  # noqa: E402
                       recombination_rate, ionized_mass, recombination_time,
                       Q_O5, Q_B0, PC, M_SUN)

Q_O9 = 5e48   # mid O9 star


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Stromgren sphere: R_s = (3 Q / 4 pi n^2 alpha_B)^(1/3)\n")
    print(f"  balance: every ionizing photon replaces one recombination\n")
    print(f"  {'star (Q, /s)':>18}{'n (/cc)':>10}{'R_s (pc)':>10}"
          f"{'M_ion (Msun)':>14}")
    print("  " + "-" * 52)
    for name, Q in (("O5 (5e49)", Q_O5), ("O9 (5e48)", Q_O9), ("B0 (1e48)", Q_B0)):
        for n_cc in (10.0, 100.0, 1000.0):
            R = stromgren_radius(Q, n_cc * 1e6)
            M = ionized_mass(R, n_cc * 1e6) / M_SUN
            print(f"  {name:>18}{n_cc:>10.0f}{R/PC:>10.2f}{M:>14.0f}")

    print("\n  R shrinks as n^(-2/3): the same O star lights up a ~25 pc bubble in")
    print("  diffuse gas but only a fraction of a parsec in a dense clump (a compact")
    print("  HII region). These are the pink emission nebulae -- Orion, the Rosette --")
    print("  that flag where massive stars formed in the last few million years.")
    tr = recombination_time(100e6) / 3.15576e7
    print(f"  Switch the star off and the bubble recombines in ~{tr:.0f} yr.")

    _svg(os.path.join(outdir, "stromgren.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'stromgren.svg')}")


def _svg(path, size=720, pad=68):
    ns = [10 ** (0.0 + 0.06 * i) for i in range(0, 67)]   # 1 .. ~1e4 /cc
    stars = [(Q_O5, "#4dabf7", "O5 (5e49)"),
             (Q_O9, "#ffd43b", "O9 (5e48)"),
             (Q_B0, "#ff6b6b", "B0 (1e48)")]
    curves = []
    for Q, col, label in stars:
        ys = [stromgren_radius_pc(Q, n) for n in ns]
        curves.append((ys, col, label))

    lx = [math.log10(n) for n in ns]
    all_y = [y for ys, _, _ in curves for y in ys]
    ly_all = [math.log10(y) for y in all_y]
    xmin, xmax = lx[0], lx[-1]
    ymin, ymax = min(ly_all), max(ly_all)

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
    for i, (ys, col, label) in enumerate(curves):
        ly = [math.log10(y) for y in ys]
        poly = " ".join(f"{sx(lx[j]):.1f},{sy(ly[j]):.1f}" for j in range(len(ns)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        parts.append(f'<text x="{size-pad-140}" y="{ytop + i*16}" fill="{col}" '
                     f'font-size="12">{label}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Stromgren radius vs cloud density</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'R ~ n^(-2/3): diffuse nebula to compact HII region</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 density (atoms/cc) -&gt;</text>')
    parts.append(f'<text x="{pad-14}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 Stromgren radius (pc)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
