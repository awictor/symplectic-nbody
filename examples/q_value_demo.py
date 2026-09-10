"""Demo: nuclear Q-values and the energy in mass.

Prints the Q-value of landmark reactions and the energy density of fuels from TNT to
matter-antimatter annihilation, then draws those energy densities on a log bar chart --
the millionfold leap from chemical to nuclear to pure mass-energy.

    python examples/q_value_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from q_value import (q_value_from_amu, energy_per_kg, mass_fraction_converted,  # noqa: E402
                     AMU, C, MEV)

DM_DT = (2.014102 + 3.016049) - (4.002602 + 1.008665)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Nuclear Q = (m_reactants - m_products) c^2  (1 amu = 931.494 MeV)\n")
    print(f"  {'reaction':>22}{'Q (MeV)':>11}{'mass converted':>17}")
    print("  " + "-" * 50)
    reactions = [
        ("D + T -> He4 + n", DM_DT, 5.03),
        ("D + D -> He3 + n", 0.00350, 4.03),
        ("p-p chain (net)", 0.02866, 4.03),
        ("U-235 fission", 0.2115, 236.0),
    ]
    for name, dm, mtot in reactions:
        Q = q_value_from_amu(dm)
        frac = dm / mtot * 100
        print(f"  {name:>22}{Q:>11.2f}{frac:>15.3f} %")

    print("\n  energy density of fuels (joules per kilogram):")
    fuels = [
        ("TNT (chemical)", 4.6e6),
        ("gasoline", 4.6e7),
        ("U-235 fission", energy_per_kg(197.0, 236.0)),
        ("D-T fusion", energy_per_kg(17.6, 5.03)),
        ("matter-antimatter", C * C),
    ]
    for name, e in fuels:
        print(f"    {name:>20}: {e:.2e} J/kg")

    print("\n  A nuclear reaction converts only a fraction of a percent of its mass, but")
    print("  c^2 makes that millions of times more than any chemical bond -- fission")
    print("  ~2 million times TNT, fusion ~4x fission. Total annihilation converts 100%")
    print("  of the mass, the ultimate energy density c^2 ~ 9x10^16 J/kg.")

    _svg(os.path.join(outdir, "q_value.svg"), fuels)
    print(f"\n  wrote {os.path.join(outdir, 'q_value.svg')}")


def _svg(path, fuels, size=720, pad=90):
    labels = [f[0] for f in fuels]
    vals = [math.log10(f[1]) for f in fuels]
    n = len(fuels)
    ymin, ymax = 5.0, 17.5
    bw = (size - 2 * pad) / n * 0.6

    def bx(i):
        return pad + (i + 0.5) * (size - 2 * pad) / n

    def by(v):
        return size - pad - (v - ymin) / (ymax - ymin) * (size - 2 * pad)

    cols = ["#8b949e", "#8b949e", "#ff922b", "#ffd43b", "#ff6b6b"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    y0 = by(ymin)
    for i, (label, v) in enumerate(zip(labels, vals)):
        x = bx(i)
        yt = by(v)
        col = cols[i % len(cols)]
        parts.append(f'<rect x="{x-bw/2:.1f}" y="{yt:.1f}" width="{bw:.1f}" '
                     f'height="{y0-yt:.1f}" fill="{col}" opacity="0.8"/>')
        parts.append(f'<text x="{x:.1f}" y="{yt-6:.1f}" fill="{col}" '
                     f'font-size="10" text-anchor="middle">1e{v:.0f}</text>')
        parts.append(f'<text x="{x:.1f}" y="{size-pad+16:.1f}" fill="#8b949e" '
                     f'font-size="9" text-anchor="middle" '
                     f'transform="rotate(20 {x:.1f} {size-pad+16:.1f})">{label}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Energy density of fuels (E = mc^2 at work)</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'chemical -&gt; nuclear -&gt; pure mass: each step a millionfold leap</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 energy per kg (J/kg)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
