"""Demo: the Bohr model of hydrogen.

Prints the energy levels and the Lyman/Balmer/Paschen series wavelengths, then draws
the energy-level diagram with the series transitions marked -- the discrete spectrum
Bohr's quantized orbits explained.

    python examples/bohr_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bohr import (energy_level_ev, radius, transition_wavelength,  # noqa: E402
                  ionization_energy_ev, orbital_speed, BOHR_RADIUS, C)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Bohr model of hydrogen: E_n = -13.6/n^2 eV, r_n = n^2 a_0\n")
    print(f"  Bohr radius a_0 = {BOHR_RADIUS*1e12:.2f} pm, "
          f"fine-structure alpha = 1/{C/orbital_speed(1):.1f}\n")
    print(f"  {'n':>4}{'E_n (eV)':>11}{'r_n (pm)':>11}")
    print("  " + "-" * 26)
    for n in range(1, 6):
        print(f"  {n:>4}{energy_level_ev(n):>11.3f}{radius(n)*1e12:>11.1f}")

    print("\n  spectral series (to lower level n1):")
    for n1, name in [(1, "Lyman (UV)"), (2, "Balmer (visible)"), (3, "Paschen (IR)")]:
        lams = ", ".join(f"{transition_wavelength(n1, n2)*1e9:.1f}"
                         for n2 in range(n1 + 1, n1 + 4))
        print(f"    {name:>18} to n={n1}:  {lams} nm ...")

    print("\n  Quantizing angular momentum (L = n hbar) forces the electron onto discrete")
    print("  orbits, so it can only emit photons of fixed energy -- the sharp lines of")
    print("  the hydrogen spectrum. H-alpha at 656 nm gives nebulae their red glow;")
    print("  the Lyman series lands in the UV, Paschen in the infrared. The model nails")
    print("  the energies exactly, and v/c at n=1 is the fine-structure constant.")

    _svg(os.path.join(outdir, "bohr.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'bohr.svg')}")


def _svg(path, size=720, pad=72):
    n_max = 6
    # energy axis: 0 at top (ionization), -13.6 at bottom
    E = [energy_level_ev(n) for n in range(1, n_max + 1)]
    emin, emax = -13.6, 0.5

    def ey(e):
        return pad + (emax - e) / (emax - emin) * (size - 2 * pad)

    x0, x1 = pad + 60, size - pad - 40

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # level lines
    for n in range(1, n_max + 1):
        y = ey(energy_level_ev(n))
        parts.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" '
                     f'stroke="#8b949e" stroke-width="1.2"/>')
        parts.append(f'<text x="{x0-8:.1f}" y="{y+4:.1f}" fill="#8b949e" '
                     f'font-size="11" text-anchor="end">n={n}</text>')
        parts.append(f'<text x="{x1+6:.1f}" y="{y+4:.1f}" fill="#484f58" '
                     f'font-size="9">{energy_level_ev(n):.2f} eV</text>')
    # ionization line
    yi = ey(0.0)
    parts.append(f'<line x1="{x0:.1f}" y1="{yi:.1f}" x2="{x1:.1f}" y2="{yi:.1f}" '
                 f'stroke="#ff6b6b" stroke-width="1" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{x0-8:.1f}" y="{yi+4:.1f}" fill="#ff6b6b" '
                 f'font-size="10" text-anchor="end">ionized</text>')

    # transitions: Lyman (to 1), Balmer (to 2), Paschen (to 3)
    series = [(1, "#b197fc", 0.0), (2, "#4dabf7", 0.28), (3, "#ff922b", 0.55)]
    for n1, col, frac in series:
        for k, n2 in enumerate(range(n1 + 1, n1 + 4)):
            xc = x0 + (0.12 + frac + 0.05 * k) * (x1 - x0)
            y_top = ey(energy_level_ev(n2))
            y_bot = ey(energy_level_ev(n1))
            parts.append(f'<line x1="{xc:.1f}" y1="{y_top:.1f}" x2="{xc:.1f}" '
                         f'y2="{y_bot:.1f}" stroke="{col}" stroke-width="1.6"/>')
            # arrowhead at bottom
            parts.append(f'<polygon points="{xc-3:.1f},{y_bot-6:.1f} {xc+3:.1f},{y_bot-6:.1f} '
                         f'{xc:.1f},{y_bot:.1f}" fill="{col}"/>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Hydrogen energy levels and spectral series</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'purple: Lyman (UV, to n=1); blue: Balmer (visible, to n=2); '
                 f'orange: Paschen (IR, to n=3)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
