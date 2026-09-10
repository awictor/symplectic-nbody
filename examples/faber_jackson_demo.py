"""Demo: the Faber-Jackson relation for elliptical galaxies.

Shows the L ~ sigma^4 scaling from dwarf to giant ellipticals, the virial mass,
and how a velocity dispersion yields a luminosity (hence a distance). Renders
luminosity vs velocity dispersion to a log-log SVG.

    python examples/faber_jackson_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from faber_jackson import (faber_jackson_luminosity, virial_mass,  # noqa: E402
                           mass_to_light, dispersion_from_luminosity,
                           M_SUN, KPC)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("The Faber-Jackson relation: L ~ sigma^4 for ellipticals\n")
    print(f"  {'sigma (km/s)':>12}{'L (L_sun)':>14}{'virial M (M_sun)':>18}")
    print("  " + "-" * 44)
    for sig_kms in (50, 100, 200, 300, 400):
        sig = sig_kms * 1e3
        L = faber_jackson_luminosity(sig)
        M = virial_mass(sig, 5 * KPC) / M_SUN
        print(f"  {sig_kms:>12}{L:>14.2e}{M:>18.2e}")
    print("\n  Luminosity climbs as the fourth power of the velocity dispersion, so")
    print("  a spectral line width (sigma) pins a galaxy's luminosity -- and,")
    print("  against its apparent brightness, its distance. It is the elliptical-")
    print("  galaxy twin of the Tully-Fisher relation for spirals.")

    _svg(os.path.join(outdir, "faber_jackson.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'faber_jackson.svg')}")


def _svg(path, size=720, pad=64):
    sigs = [10 ** (1.5 + 0.012 * i) for i in range(0, 101)]  # ~30 .. ~500 km/s (in km/s)
    Ls = [faber_jackson_luminosity(s * 1e3) for s in sigs]
    ls = [math.log10(s) for s in sigs]
    lL = [math.log10(L) for L in Ls]
    smin, smax = ls[0], ls[-1]
    Lmin, Lmax = lL[0], lL[-1]

    def sx(x):
        return pad + (x - smin) / (smax - smin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - Lmin) / (Lmax - Lmin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(ls[i]):.1f},{sy(lL[i]):.1f}" for i in range(len(sigs)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#e63946" stroke-width="2.2"/>')
    # mark L* (200 km/s)
    l200 = math.log10(200.0)
    parts.append(f'<circle cx="{sx(l200):.1f}" cy="{sy(math.log10(2e10)):.1f}" r="4" fill="#e9c46a"/>')
    parts.append(f'<text x="{sx(l200)+8:.1f}" y="{sy(math.log10(2e10)):.1f}" fill="#e9c46a" '
                 f'font-size="11">L* (200 km/s)</text>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Faber-Jackson: L ~ sigma^4 (log-log)</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 velocity dispersion (km/s) -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 L / L_sun</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
