"""Demo: the Hubble diagram and the discovery of cosmic acceleration.

Compares the distance modulus mu(z) of a dark-energy universe (LCDM) with a
decelerating one (Einstein-de Sitter). Distant supernovae sit above the EdS
curve -- fainter than a decelerating universe predicts -- which is how dark
energy was found. Also shows the angular-diameter-distance turnover. Renders both
to SVG.

    python examples/distances_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from friedmann import Cosmology  # noqa: E402
from distances import (distance_modulus, angular_diameter_distance,  # noqa: E402
                       angular_diameter_peak)

LCDM = Cosmology(Omega_m=0.3, Omega_L=0.7)
EDS = Cosmology(Omega_m=1.0, Omega_L=0.0)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Cosmic distances: the Hubble diagram and cosmic acceleration\n")
    print(f"  {'z':>5}{'mu LCDM':>12}{'mu EdS':>12}{'Delta mu (fainter)':>22}")
    print("  " + "-" * 51)
    for z in (0.2, 0.5, 1.0, 1.5):
        mu_l = distance_modulus(LCDM, z)
        mu_e = distance_modulus(EDS, z)
        print(f"  {z:>5.1f}{mu_l:>12.3f}{mu_e:>12.3f}{mu_l - mu_e:>+22.3f}")
    print(f"\n  Type-Ia supernovae at z~0.5 sit ~0.4 mag ABOVE the decelerating")
    print(f"  prediction -- fainter, hence farther, hence the expansion is")
    print(f"  accelerating. That is the 1998 result (2011 Nobel Prize).\n")
    print(f"  angular-diameter distance peaks at z = {angular_diameter_peak(LCDM):.2f}")
    print(f"  (past it, more distant objects look angularly LARGER -- why the CMB")
    print(f"  acoustic spots subtend about a degree).")

    zs = [0.02 + 3.0 * i / 199 for i in range(200)]
    mu_l = [distance_modulus(LCDM, z) for z in zs]
    mu_e = [distance_modulus(EDS, z) for z in zs]
    da_l = [angular_diameter_distance(LCDM, z) for z in zs]
    _svg(zs, mu_l, mu_e, da_l, os.path.join(outdir, "distances.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'distances.svg')}")


def _svg(zs, mu_l, mu_e, da_l, path, size=720, pad=56):
    zmax = zs[-1]
    mu_lo = min(min(mu_l), min(mu_e))
    mu_hi = max(max(mu_l), max(mu_e))
    da_hi = max(da_l)

    def sx(z):
        return pad + z / zmax * (size - 2 * pad)

    def sy_mu(mu):
        return size - pad - (mu - mu_lo) / (mu_hi - mu_lo) * (size - 2 * pad)

    def sy_da(d):
        return size - pad - d / (da_hi * 1.1) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    for vals, sy, col, label in ((mu_l, sy_mu, "#4cc9f0", "mu(z): LCDM (dark energy)"),
                                 (mu_e, sy_mu, "#e63946", "mu(z): Einstein-de Sitter"),
                                 (da_l, sy_da, "#f4a261", "D_A(z): LCDM (turns over)")):
        poly = " ".join(f"{sx(zs[i]):.1f},{sy(vals[i]):.1f}" for i in range(len(zs)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="1.8"/>')
    y = pad + 8
    for col, label in (("#4cc9f0", "distance modulus, LCDM"),
                       ("#e63946", "distance modulus, EdS (fainter gap = dark energy)"),
                       ("#f4a261", "angular-diameter distance, LCDM (turns over)")):
        parts.append(f'<text x="{pad+14}" y="{y}" fill="{col}" font-size="12">{label}</text>')
        y += 16
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Hubble diagram: dark energy makes distant SNe fainter</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">redshift z -&gt;</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
