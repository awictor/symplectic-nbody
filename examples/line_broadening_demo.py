"""Demo: spectral line broadening mechanisms.

Prints Doppler, natural and pressure widths of H-alpha across environments and names
the dominant mechanism, then draws a hot Doppler-broadened Gaussian line beside a
dense pressure-broadened Lorentzian line for comparison.

    python examples/line_broadening_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from line_broadening import (thermal_speed, doppler_width,  # noqa: E402
                             doppler_width_wavelength, natural_width,
                             pressure_width, dominant_mechanism, AMU, C)

LAM = 656.3e-9
NU0 = C / LAM
M_H = 1.008 * AMU
A_HALPHA = 6.5e7   # Einstein A for H-alpha


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Line broadening of H-alpha (656.3 nm): Doppler / natural / pressure\n")
    print(f"  natural width (fixed): {natural_width(A_HALPHA)/1e6:.1f} MHz\n")
    print(f"  {'environment':>22}{'T (K)':>9}{'Doppler':>12}{'pressure':>12}"
          f"{'dominant':>12}")
    print("  " + "-" * 67)
    # (name, T, collision rate)
    envs = [
        ("HII region", 1e4, 1e4),
        ("solar photosphere", 6000.0, 1e9),
        ("red giant", 4000.0, 1e7),
        ("white-dwarf atmos", 1e4, 1e13),
        ("cool ISM cloud", 100.0, 1e2),
    ]
    for name, T, col in envs:
        dD = doppler_width(NU0, T, M_H)
        dP = pressure_width(col)
        dom = dominant_mechanism(NU0, T, M_H, A_HALPHA, col)
        print(f"  {name:>22}{T:>9.0f}{dD/1e9:>10.2f}G{dP/1e9:>11.3f}G{dom:>12}")

    print("\n  Doppler width ~ sqrt(T/m) reads the temperature (and, from the ratio of")
    print("  species widths, tells light atoms from heavy). Pressure width grows with")
    print("  density, so a dense white-dwarf photosphere has hugely broadened,")
    print("  Lorentzian-winged lines while a thin HII region is purely Doppler. The")
    print("  observed shape is the Voigt convolution of the Gaussian and Lorentzian.")

    _svg(os.path.join(outdir, "line_broadening.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'line_broadening.svg')}")


def _gauss(x, w):
    return math.exp(-(x / w) ** 2)


def _lorentz(x, w):
    return 1.0 / (1.0 + (x / w) ** 2)


def _svg(path, size=720, pad=70):
    # frequency offset axis in GHz, both profiles normalized to peak 1
    xs = [-40.0 + 0.5 * i for i in range(0, 161)]   # -40..+40 GHz
    wD = doppler_width(NU0, 1e4, M_H) / 1e9         # Doppler half-width, GHz
    wL = pressure_width(1e12) / 1e9                 # a broad pressure width, GHz
    gy = [_gauss(x, wD) for x in xs]
    ly = [_lorentz(x, wL) for x in xs]
    xmin, xmax = xs[0], xs[-1]

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - y * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    gp = " ".join(f"{sx(xs[i]):.1f},{sy(gy[i]):.1f}" for i in range(len(xs)))
    lp = " ".join(f"{sx(xs[i]):.1f},{sy(ly[i]):.1f}" for i in range(len(xs)))
    parts.append(f'<polyline points="{gp}" fill="none" stroke="#4dabf7" stroke-width="2.4"/>')
    parts.append(f'<polyline points="{lp}" fill="none" stroke="#ff922b" stroke-width="2.4"/>')

    parts.append(f'<text x="{pad+10}" y="{pad+22}" fill="#4dabf7" font-size="12">'
                 f'Doppler (Gaussian, thermal)</text>')
    parts.append(f'<text x="{pad+10}" y="{pad+38}" fill="#ff922b" font-size="12">'
                 f'pressure (Lorentzian, dense) -- broad wings</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Line profiles: Gaussian core vs Lorentzian wings</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'thermal motion gives a Gaussian; collisions give broad Lorentzian wings</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">frequency offset from line centre (GHz) -&gt;</text>')
    parts.append(f'<text x="{pad-16}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'normalized intensity</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
