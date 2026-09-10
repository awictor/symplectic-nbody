"""Demo: the Tully-Fisher relation.

Prints luminosity, absolute magnitude and baryonic mass across a range of spiral
rotation speeds, then draws the log L vs log v_flat line with its slope-4 power law and
several galaxies plotted -- the relation that turns a rotation width into a distance.

    python examples/tully_fisher_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tully_fisher import (luminosity_from_vflat, absolute_magnitude,  # noqa: E402
                          baryonic_mass_from_vflat, distance_from_apparent)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Tully-Fisher: L ~ v_flat^4 -- spirals that spin faster shine brighter\n")
    print(f"  {'galaxy type':>18}{'v_flat':>9}{'L (Lsun)':>12}{'M_abs':>9}"
          f"{'M_baryon':>12}")
    print("  " + "-" * 60)
    rows = [
        ("dwarf spiral", 80.0),
        ("small spiral", 120.0),
        ("Milky Way-like", 220.0),
        ("massive spiral", 300.0),
        ("giant spiral", 400.0),
    ]
    for name, v in rows:
        L = luminosity_from_vflat(v)
        print(f"  {name:>18}{v:>9.0f}{L:>12.2e}{absolute_magnitude(v):>9.2f}"
              f"{baryonic_mass_from_vflat(v):>12.2e}")

    print("\n  A four-fold jump in luminosity for every doubling of rotation speed --")
    print("  because v^2 = GM/R and spirals hold roughly constant surface brightness,")
    print("  so mass, spin and light rise together. Since the rotation width is easy")
    print("  to measure (the 21-cm line), Tully-Fisher gives redshift-independent")
    print("  distances far beyond where individual Cepheids can be resolved.")

    _svg(os.path.join(outdir, "tully_fisher.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'tully_fisher.svg')}")


def _svg(path, size=720, pad=72):
    vs = [60.0 * (1.03 ** i) for i in range(0, 70)]   # ~60 .. ~460 km/s
    Ls = [luminosity_from_vflat(v) for v in vs]
    lx = [math.log10(v) for v in vs]
    ly = [math.log10(L) for L in Ls]
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
    poly = " ".join(f"{sx(lx[i]):.1f},{sy(ly[i]):.1f}" for i in range(len(vs)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#8338ec" stroke-width="2.6"/>')

    for name, v, col in [("dwarf", 80.0, "#4dabf7"), ("Milky Way", 220.0, "#ffd43b"),
                         ("giant", 400.0, "#ff6b6b")]:
        px = sx(math.log10(v))
        py = sy(math.log10(luminosity_from_vflat(v)))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{col}"/>')
        parts.append(f'<text x="{px+8:.1f}" y="{py+4:.1f}" fill="{col}" '
                     f'font-size="11">{name}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Tully-Fisher: luminosity vs rotation speed</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'slope 4 on a log-log plot: L ~ v_flat^4</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 flat rotation speed (km/s) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 luminosity (L_sun)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
