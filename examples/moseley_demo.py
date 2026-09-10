"""Demo: Moseley's law -- ordering the elements by X-ray colour.

Prints the K-alpha energy and wavelength for a range of elements and shows how a measured
line identifies an element, then draws the Moseley plot: sqrt(K-alpha frequency) versus atomic
number, the straight line that put the periodic table in order of nuclear charge.

    python examples/moseley_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moseley import (k_alpha_energy_ev, k_alpha_wavelength, sqrt_frequency,  # noqa: E402
                     atomic_number_from_energy)


ELEMENTS = [(20, "Ca"), (26, "Fe"), (29, "Cu"), (42, "Mo"), (47, "Ag"), (74, "W")]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Moseley's law: sqrt(f) = a(Z - 1) -- X-ray lines order the periodic table\n")
    print(f"  {'element':<10}{'Z':>4}{'K-alpha energy':>18}{'wavelength':>14}")
    for z, sym in ELEMENTS:
        E = k_alpha_energy_ev(z)
        lam = k_alpha_wavelength(z)
        print(f"  {sym:<10}{z:>4}{E/1000:>14.2f} keV{lam*1e9:>11.4f} nm")

    print("\n  Identifying an unknown from its K-alpha line:")
    for E, truth in ((6380.0, "Fe"), (8000.0, "Cu"), (21590.0, "Ag")):
        z = atomic_number_from_energy(E)
        print(f"    line at {E/1000:>5.1f} keV  ->  Z = {round(z)} ({truth})")

    print("\n  Because sqrt(f) is exactly linear in Z, Moseley's plot ordered the elements by")
    print("  nuclear charge -- not atomic weight -- and its gaps predicted where undiscovered")
    print("  elements (technetium, promethium) had to be. Still the basis of XRF elemental ID.")

    _svg(os.path.join(outdir, "moseley.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'moseley.svg')}")


def _svg(path, size=720, pad=80):
    z_min, z_max = 10, 80
    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 44

    s_max = sqrt_frequency(z_max) * 1.05

    def X(z):
        return x0 + (z - z_min) / (z_max - z_min) * (x1 - x0)

    def Y(s):
        return y0 - s / s_max * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Moseley&#39;s law</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'sqrt(K-alpha frequency) is a straight line in atomic number Z -- the periodic table&#39;s true order</text>',
    ]

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.5"/>')
    for z in range(10, 81, 10):
        parts.append(f'<text x="{X(z):.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{z}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+32:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">atomic number Z</text>')
    parts.append(f'<text x="24" y="{(y0+y1)/2:.1f}" fill="#8b949e" font-size="11" '
                 f'transform="rotate(-90 24 {(y0+y1)/2:.1f})" text-anchor="middle">sqrt(K-alpha frequency)</text>')

    # the straight Moseley line
    pts = " ".join(f"{X(z):.1f},{Y(sqrt_frequency(z)):.1f}" for z in range(z_min, z_max + 1))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')

    # mark the tabulated elements
    for z, sym in ELEMENTS:
        parts.append(f'<circle cx="{X(z):.1f}" cy="{Y(sqrt_frequency(z)):.1f}" r="5" fill="#ffd43b"/>')
        parts.append(f'<text x="{X(z)+7:.1f}" y="{Y(sqrt_frequency(z))+4:.1f}" fill="#ffd43b" '
                     f'font-size="11">{sym} ({z})</text>')

    parts.append(f'<text x="{X(20):.1f}" y="{Y(sqrt_frequency(60)):.1f}" fill="#8b949e" '
                 f'font-size="11">straight line: gaps meant missing elements</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
