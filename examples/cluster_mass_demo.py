"""Demo: weighing galaxy clusters from their velocity dispersion.

Prints the virial mass, escape velocity and mass-to-light ratio of real clusters, then draws
the mass-to-light ladder: from a single star through a galaxy to a cluster, the M/L jumps by
orders of magnitude -- Zwicky's 1933 evidence for dark matter.

    python examples/cluster_mass_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cluster_mass import (virial_mass_los, escape_velocity, mass_to_light,  # noqa: E402
                          dark_matter_fraction, M_SUN, MPC)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Weighing clusters by their motion: M = alpha sigma^2 R / G (virial)\n")
    print(f"  {'cluster':<16}{'sigma_los':>12}{'R (Mpc)':>10}{'M (Msun)':>13}{'M/L':>8}")
    # (name, sigma_los km/s, R Mpc, luminosity in 1e12 Lsun)
    clusters = [
        ("Virgo", 700.0, 1.5, 3.0),
        ("Coma", 1000.0, 1.5, 5.0),
        ("Norma", 900.0, 1.3, 4.0),
        ("Bullet", 1200.0, 2.0, 6.0),
    ]
    for name, s, R, L12 in clusters:
        M = virial_mass_los(s, R)
        ml = mass_to_light(M, L12 * 1e12)
        print(f"  {name:<16}{s:>9.0f} km/s{R:>10.1f}{M/M_SUN:>13.2e}{ml:>8.0f}")

    print("\n  Escape velocity from Coma (1e15 Msun, 1.5 Mpc): %.0f km/s -- its galaxies at"
          % (escape_velocity(1e15 * M_SUN, 1.5 * MPC) / 1000))
    print("  ~1000 km/s would fly apart on stellar mass alone. Zwicky (1933) found the mass")
    print("  needed to bind Coma was ~100x its visible stars: the first case for dark matter.")

    fdm = dark_matter_fraction(1e15 * M_SUN, 5e13 * M_SUN)
    print("  With only ~5%% of the mass luminous, the dark-matter fraction is ~%.0f%%." % (fdm * 100))

    _svg(os.path.join(outdir, "cluster_mass.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'cluster_mass.svg')}")


def _svg(path, size=720, pad=80):
    # Mass-to-light ladder on a log axis: star, galaxy disk, cluster.
    systems = [
        ("Sun (star)", 1.0, "#ffd43b"),
        ("stellar population", 3.0, "#ff922b"),
        ("spiral galaxy (+halo)", 30.0, "#06d6a0"),
        ("galaxy cluster", 300.0, "#4dabf7"),
    ]
    ml_max = 1000.0

    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 60
    n = len(systems)
    bw = (x1 - x0) / n * 0.55

    lo, hi = math.log10(1.0), math.log10(ml_max)

    def Y(ml):
        return y0 - (math.log10(ml) - lo) / (hi - lo) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The mass-to-light ladder</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'dynamical M/L climbs from a few (stars) to hundreds (clusters) -- the dark-matter signal</text>',
    ]

    # axis + decade gridlines
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.5"/>')
    for e in range(0, 4):
        ml = 10.0 ** e
        gy = Y(ml)
        parts.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{x0-8:.1f}" y="{gy+4:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="end">{int(ml)}</text>')
    parts.append(f'<text x="24" y="{(y0+y1)/2:.1f}" fill="#8b949e" font-size="11" '
                 f'transform="rotate(-90 24 {(y0+y1)/2:.1f})" text-anchor="middle">M/L (solar units)</text>')

    # "stars only" band (M/L ~ 1-5)
    parts.append(f'<rect x="{x0:.1f}" y="{Y(5.0):.1f}" width="{x1-x0:.1f}" '
                 f'height="{Y(1.0)-Y(5.0):.1f}" fill="#8b949e" opacity="0.12"/>')
    parts.append(f'<text x="{x1-4:.1f}" y="{Y(4.0)-4:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end">stars alone</text>')

    for i, (label, ml, col) in enumerate(systems):
        cx = x0 + (x1 - x0) * (i + 0.5) / n
        top = Y(ml)
        parts.append(f'<rect x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                     f'height="{y0-top:.1f}" fill="{col}" opacity="0.85"/>')
        parts.append(f'<text x="{cx:.1f}" y="{top-8:.1f}" fill="{col}" font-size="12" '
                     f'text-anchor="middle">{ml:g}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{label}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
