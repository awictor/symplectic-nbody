"""Demo: the CMB acoustic scale -- why the microwave-background spots are ~1 deg.

Computes the sound horizon at recombination, the distance to the last-scattering
surface, the acoustic angle, and the first-peak multipole (~220), then renders a
schematic acoustic-peak comb to SVG.

    python examples/cmb_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from friedmann import Cosmology  # noqa: E402
from saha import recombination_redshift  # noqa: E402
from cmb import (sound_horizon, acoustic_angle, first_peak_multipole)  # noqa: E402
from distances import comoving_distance  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    cosmo = Cosmology(Omega_m=0.3, Omega_L=0.7)
    z_rec = recombination_redshift(0.5)
    l1 = first_peak_multipole(cosmo, z_rec)

    print("The CMB acoustic scale (flat LCDM)\n")
    print(f"  recombination redshift    : z = {z_rec:.0f}")
    print(f"  sound horizon r_s         : {sound_horizon(cosmo, z_rec):.0f} Mpc")
    print(f"  distance to last scattering: {comoving_distance(cosmo, z_rec)*299792.458/70:.0f} Mpc")
    print(f"  acoustic angle theta      : {math.degrees(acoustic_angle(cosmo, z_rec)):.2f} deg")
    print(f"  first acoustic peak       : l ~ {l1:.0f}  (WMAP/Planck: 220)\n")
    print("  A fixed sound-horizon ruler at the edge of the observable universe")
    print("  subtends about a degree; the harmonics of that standing wave are the")
    print("  acoustic peaks, and their position pins the geometry to flat.")

    _svg(l1, os.path.join(outdir, "cmb.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'cmb.svg')}")


def _svg(l1, path, size=720, pad=64):
    # schematic power spectrum: damped harmonic peaks at l ~ n * l1
    lmax = 5.0 * l1

    def sx(l):
        return pad + l / lmax * (size - 2 * pad)

    def sy(p):
        return size - pad - p * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    # schematic C_l: sum of damped gaussian peaks at n*l1
    pts = []
    ls = [lmax * i / 400 for i in range(1, 401)]
    for l in ls:
        p = 0.0
        for n in range(1, 6):
            center = n * l1
            width = 0.35 * l1
            amp = math.exp(-0.5 * ((n - 1) / 2.0))  # damping of higher peaks
            p += amp * math.exp(-0.5 * ((l - center) / width) ** 2)
        pts.append((l, p))
    pmax = max(p for _l, p in pts)
    poly = " ".join(f"{sx(l):.1f},{sy(p/pmax):.1f}" for l, p in pts)
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4cc9f0" stroke-width="2.2"/>')
    parts.append(f'<line x1="{sx(l1):.1f}" y1="{pad}" x2="{sx(l1):.1f}" y2="{size-pad}" '
                 f'stroke="#e63946" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{sx(l1)+6:.1f}" y="{pad+16}" fill="#e63946" font-size="12">'
                 f'first peak l~{l1:.0f}</text>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'CMB acoustic peaks (schematic power spectrum)</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'multipole l (angular scale ~ 180/l degrees) -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'temperature power (arb.)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
