"""Demo: MOND -- flat rotation curves and Tully-Fisher without dark matter.

Contrasts the MOND rotation curve of a bare baryonic mass (which flattens on its
own) with the Newtonian Keplerian decline, and shows the baryonic Tully-Fisher
relation v^4 ~ M. Renders both to SVG.

    python examples/mond_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mond import (circular_speed, deep_mond_vflat, tully_fisher_mass,  # noqa: E402
                  G, A0, M_SUN, KM, KPC)


def newton_speed(M, r):
    return math.sqrt(G * M / r)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    M = 6e10 * M_SUN
    print("MOND: modified gravity instead of dark matter\n")
    print(f"  a0 = {A0:.2e} m/s^2   baryonic mass = 6e10 M_sun")
    print(f"  asymptotic flat speed v = (G M a0)^1/4 = {deep_mond_vflat(M)/KM:.1f} km/s\n")
    print(f"  {'r (kpc)':>8}{'Newton (km/s)':>16}{'MOND (km/s)':>14}")
    print("  " + "-" * 38)
    curve = []
    for r_kpc in (2, 5, 10, 20, 40, 80):
        r = r_kpc * KPC
        vn = newton_speed(M, r) / KM
        vm = circular_speed(M, r) / KM
        curve.append((r_kpc, vn, vm))
        print(f"  {r_kpc:>8}{vn:>16.1f}{vm:>14.1f}")
    print("\n  Newtonian gravity on the visible mass alone falls off (Keplerian);")
    print("  MOND flattens the curve with no dark matter. Its sharpest prediction is")
    print("  the baryonic Tully-Fisher law v^4 = G M a0, tight across real galaxies.")

    _svg(curve, M, os.path.join(outdir, "mond.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'mond.svg')}")


def _svg(curve, M, path, size=720, pad=64):
    rmax = curve[-1][0] * 1.05
    vmax = max(max(vn, vm) for _r, vn, vm in curve) * 1.1

    def sx(r):
        return pad + r / rmax * (size - 2 * pad)

    def sy(v):
        return size - pad - v / vmax * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    vflat = deep_mond_vflat(M) / KM
    parts.append(f'<line x1="{pad}" y1="{sy(vflat):.1f}" x2="{size-pad}" y2="{sy(vflat):.1f}" '
                 f'stroke="#8b949e" stroke-dasharray="4,4"/>')
    npoly = " ".join(f"{sx(r):.1f},{sy(vn):.1f}" for r, vn, _vm in curve)
    mpoly = " ".join(f"{sx(r):.1f},{sy(vm):.1f}" for r, _vn, vm in curve)
    parts.append(f'<polyline points="{npoly}" fill="none" stroke="#e63946" stroke-width="2"/>')
    parts.append(f'<polyline points="{mpoly}" fill="none" stroke="#4cc9f0" stroke-width="2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'MOND (blue, flat) vs Newton on visible mass (red, declining)</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+8}" fill="#4cc9f0" font-size="12">'
                 f'MOND: modified gravity, no dark matter</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+26}" fill="#e63946" font-size="12">'
                 f'Newton (baryons only): Keplerian decline</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">radius (kpc) -&gt;</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
