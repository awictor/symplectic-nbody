"""Demo: Compton and inverse-Compton scattering.

Shows the Compton wavelength shift and scattered-photon energy vs scattering
angle, and the inverse-Compton energy boost that turns CMB photons into X-rays.
Renders the scattered energy vs angle to SVG.

    python examples/compton_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compton import (compton_shift, scattered_energy, inverse_compton_boost,  # noqa: E402
                     electron_rest_energy_kev, LAMBDA_C, KEV)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Compton scattering (photon off a stationary electron)\n")
    print(f"  Compton wavelength   : {LAMBDA_C*1e12:.3f} pm")
    print(f"  electron rest energy : {electron_rest_energy_kev():.0f} keV\n")
    E0 = 500 * KEV
    print(f"  {'angle (deg)':>12}{'shift (pm)':>12}{'E scattered (keV)':>20}")
    print("  " + "-" * 44)
    for deg in (0, 45, 90, 135, 180):
        th = math.radians(deg)
        print(f"  {deg:>12}{compton_shift(th)*1e12:>12.3f}"
              f"{scattered_energy(E0, th)/KEV:>20.1f}")
    print("\n  Inverse Compton (fast electron kicks a photon UP):")
    for g in (10, 100, 1000):
        print(f"    gamma={g:>5}: boost x{inverse_compton_boost(g):.0f}  "
              f"(a 1 meV CMB photon -> {1e-3*inverse_compton_boost(g):.0f} eV)")
    print("\n  Compton down-shifts photons off electrons; inverse Compton up-shifts")
    print("  them off relativistic electrons -- how CMB and starlight become X-rays.")

    _svg(E0, os.path.join(outdir, "compton.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'compton.svg')}")


def _svg(E0, path, size=720, pad=64):
    degs = [d for d in range(0, 181)]
    Es = [scattered_energy(E0, math.radians(d)) / KEV for d in degs]
    Emin, Emax = min(Es), max(Es)

    def sx(d):
        return pad + d / 180.0 * (size - 2 * pad)

    def sy(E):
        return size - pad - (E - Emin) / (Emax - Emin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(degs[i]):.1f},{sy(Es[i]):.1f}" for i in range(len(degs)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4cc9f0" stroke-width="2.2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Compton: scattered photon energy vs angle (500 keV in)</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'scattering angle (deg); most energy lost at back-scatter (180)</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'E scattered (keV)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
