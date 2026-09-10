"""Demo: blackbody radiation -- Planck spectra, Wien's peak, Stefan-Boltzmann.

Shows the peak wavelength for objects from the CMB to a hot star (Wien's law) and
the T^4 luminosity scaling, then renders normalized Planck spectra at several
temperatures to SVG.

    python examples/blackbody_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from blackbody import (planck_wavelength, wien_peak_wavelength,  # noqa: E402
                       stefan_boltzmann_flux)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Blackbody radiation: Wien's peak and Stefan-Boltzmann\n")
    print(f"  {'object':<20}{'T (K)':>10}{'peak':>14}{'flux (W/m^2)':>16}")
    print("  " + "-" * 60)
    cases = [("CMB", 2.725), ("human body", 310), ("cool star (M)", 3000),
             ("Sun (G)", 5772), ("hot star (B)", 20000)]
    for name, T in cases:
        lam = wien_peak_wavelength(T)
        unit = f"{lam*1e9:.0f} nm" if lam < 1e-5 else f"{lam*1e6:.1f} um" if lam < 1e-3 else f"{lam*1e3:.2f} mm"
        print(f"  {name:<20}{T:>10.0f}{unit:>14}{stefan_boltzmann_flux(T):>16.2e}")
    print("\n  Hotter bodies peak bluer (Wien) and radiate vastly more (Stefan-")
    print("  Boltzmann T^4): a 20000 K B-star outshines the Sun per unit area by")
    print(f"  {(stefan_boltzmann_flux(20000)/stefan_boltzmann_flux(5772)):.0f}x. The 2.725 K CMB peaks in the microwave.")

    _svg([3000, 5772, 20000], os.path.join(outdir, "blackbody.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'blackbody.svg')}")


def _svg(temps, path, size=720, pad=64):
    # normalized Planck spectra vs wavelength (nm), 100 nm .. 2000 nm
    lams = [100e-9 + (2000e-9 - 100e-9) * i / 300 for i in range(301)]
    colors = {3000: "#e63946", 5772: "#f4a261", 20000: "#4cc9f0"}
    curves = []
    gmax = 0.0
    for T in temps:
        B = [planck_wavelength(l, T) for l in lams]
        curves.append((T, B))
        gmax = max(gmax, max(B))

    lam_lo, lam_hi = lams[0] * 1e9, lams[-1] * 1e9

    def sx(lam_nm):
        return pad + (lam_nm - lam_lo) / (lam_hi - lam_lo) * (size - 2 * pad)

    def sy(B):
        return size - pad - B / gmax * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
        # visible band marker (400-700 nm)
        f'<rect x="{sx(400):.1f}" y="{pad}" width="{sx(700)-sx(400):.1f}" '
        f'height="{size-2*pad}" fill="#ffffff" fill-opacity="0.05"/>',
    ]
    y = pad + 8
    for T, B in curves:
        col = colors.get(T, "#e6edf3")
        poly = " ".join(f"{sx(lams[i]*1e9):.1f},{sy(B[i]):.1f}" for i in range(len(lams)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2"/>')
        parts.append(f'<text x="{size-pad-8}" y="{y}" fill="{col}" font-size="12" '
                     f'text-anchor="end">{T} K</text>')
        y += 16
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Planck spectra (hotter = bluer &amp; brighter)</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'wavelength (nm); shaded = visible band</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
