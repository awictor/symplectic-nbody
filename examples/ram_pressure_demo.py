"""Demo: ram-pressure stripping of galaxies in clusters.

Prints the surviving gas radius for a Milky-Way-like disk across environments from a
poor group to a rich cluster core, then draws the stripping radius vs infall speed for
several ICM densities -- how the harshest environments strip a galaxy to its core.

    python examples/ram_pressure_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ram_pressure import (ram_pressure, icm_density, restoring_pressure,  # noqa: E402
                          is_stripped, stripping_radius, M_SUN, KPC)

PC = KPC / 1000.0
SIGMA0_STAR = 800.0 * M_SUN / PC ** 2
SIGMA0_GAS = 40.0 * M_SUN / PC ** 2
H = 3.0 * KPC


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Ram-pressure stripping (Gunn-Gott): rho_icm v^2 > 2 pi G Sigma_s Sigma_g\n")
    print("  Milky-Way-like disk (scale length 3 kpc); gas kept inside R_strip:\n")
    print(f"  {'environment':>22}{'n (/cc)':>10}{'v (km/s)':>10}"
          f"{'R_strip (kpc)':>15}")
    print("  " + "-" * 57)
    envs = [
        ("field / isolated", 1e-5, 300),
        ("poor group", 1e-4, 500),
        ("cluster outskirts", 5e-4, 1000),
        ("rich cluster", 1e-3, 1500),
        ("dense core, fast", 3e-3, 2000),
    ]
    for name, n, v_km in envs:
        rho = icm_density(n)
        R = stripping_radius(rho, v_km * 1e3, SIGMA0_STAR, SIGMA0_GAS, H) / KPC
        rstr = f"{R:.2f}" if R > 0 else "0 (all gone)"
        print(f"  {name:>22}{n:>10.0e}{v_km:>10.0f}{rstr:>15}")

    print("\n  A disk keeps only the gas inside R_strip, where its self-gravity still")
    print("  beats the cluster wind. In a rich cluster a Milky-Way-like galaxy is")
    print("  stripped down to a few kpc on a single pass -- its star formation")
    print("  quenches from the outside in, helping turn infalling spirals into the")
    print("  gas-poor S0s and ellipticals that crowd cluster cores.")

    _svg(os.path.join(outdir, "ram_pressure.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'ram_pressure.svg')}")


def _svg(path, size=720, pad=68):
    vs = [200.0 + 40.0 * i for i in range(0, 50)]   # 200 .. ~2200 km/s
    dens = [(1e-4, "#06d6a0", "n=1e-4 (group)"),
            (5e-4, "#ffd43b", "n=5e-4"),
            (1e-3, "#ff922b", "n=1e-3 (cluster)"),
            (3e-3, "#ff6b6b", "n=3e-3 (core)")]
    curves = []
    for n, col, label in dens:
        rho = icm_density(n)
        ys = [stripping_radius(rho, v * 1e3, SIGMA0_STAR, SIGMA0_GAS, H) / KPC
              for v in vs]
        curves.append((ys, col, label))

    xmin, xmax = vs[0], vs[-1]
    ymax = max(y for ys, _, _ in curves for y in ys) * 1.05
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
    ytop = pad + 22
    for i, (ys, col, label) in enumerate(curves):
        poly = " ".join(f"{sx(vs[j]):.1f},{sy(ys[j]):.1f}" for j in range(len(vs)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2.3"/>')
        parts.append(f'<text x="{size-pad-160}" y="{ytop + i*16}" fill="{col}" '
                     f'font-size="12">{label}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Ram-pressure stripping radius vs infall speed</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'faster, denser environments strip a galaxy to its dense core</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">infall speed (km/s) -&gt;</text>')
    parts.append(f'<text x="{pad-14}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'surviving gas radius (kpc)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
