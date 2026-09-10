"""Demo: the Penrose process -- mining a black hole's spin energy.

Shows how much of a black hole's mass-energy is extractable rotational energy as
a function of spin (up to 29% for an extremal hole), and how removing that spin
grows the irreducible mass and horizon area (the area theorem). Renders the
extractable-fraction curve to SVG.

    python examples/penrose_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from penrose import (irreducible_mass, rotational_energy_fraction,  # noqa: E402
                     max_efficiency_extremal, area_irreducible)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("The Penrose process: extracting a black hole's rotational energy\n")
    print(f"  maximum extractable fraction (extremal a=M): "
          f"{max_efficiency_extremal()*100:.1f}% of M c^2\n")
    print(f"  {'a/M':>6}{'M_irr/M':>10}{'E_rot fraction':>16}{'horizon area':>14}")
    print("  " + "-" * 46)
    for a in (0.0, 0.3, 0.6, 0.9, 0.99, 1.0):
        print(f"  {a:>6.2f}{irreducible_mass(1.0, a):>10.4f}"
              f"{rotational_energy_fraction(1.0, a)*100:>15.1f}%{area_irreducible(1.0, a):>14.2f}")
    print("\n  Up to ~29% of an extremal hole's mass-energy can be mined from its")
    print("  spin. Doing so raises the irreducible mass and horizon area -- the")
    print("  area never shrinks (Hawking), which is the second law for black holes.")

    _svg(os.path.join(outdir, "penrose.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'penrose.svg')}")


def _svg(path, size=720, pad=64):
    aa = [0.001 * i for i in range(0, 1001)]
    frac = [rotational_energy_fraction(1.0, a) * 100 for a in aa]
    fmax = max(frac)

    def sx(a):
        return pad + a * (size - 2 * pad)

    def sy(f):
        return size - pad - f / (fmax * 1.1) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{sy(fmax):.1f}" x2="{size-pad}" y2="{sy(fmax):.1f}" '
        f'stroke="#e63946" stroke-dasharray="5,4"/>',
        f'<text x="{size-pad-4}" y="{sy(fmax)-6:.1f}" fill="#e63946" font-size="12" '
        f'text-anchor="end">max 29.3% (extremal)</text>',
    ]
    poly = " ".join(f"{sx(aa[i]):.1f},{sy(frac[i]):.1f}" for i in range(len(aa)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#8338ec" stroke-width="2.2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Extractable rotational energy vs black-hole spin</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'spin a/M; extraction grows the irreducible mass (area theorem)</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'extractable fraction of M c^2 (%)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
