"""Demo: the Stark effect -- electric fields splitting and ionizing atoms.

Prints the linear Stark splitting of hydrogen levels, the quadratic ground-state shift, and
the field-ionization threshold versus principal quantum number, then draws the linear Stark
fan (levels spreading with field) and the plummeting ionization field for Rydberg atoms.

    python examples/stark_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stark import (linear_stark_pattern, quadratic_stark_shift,  # noqa: E402
                   field_ionization_threshold, hydrogen_binding_energy, RYDBERG_J)

EV = 1.602176634e-19


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Stark effect: an electric field splits (hydrogen: linearly) and ionizes atoms\n")
    E = 5e6   # 5 MV/m
    print("  Linear Stark components of hydrogen at 5 MV/m (energy shift, ueV):")
    for n in (2, 3, 4):
        comps = linear_stark_pattern(n, E)
        s = ", ".join(f"{c/EV*1e6:+.1f}" for c in comps)
        print(f"    n={n}: {len(comps)} lines  [{s}]")

    print("\n  Field-ionization threshold vs principal quantum number:")
    print(f"  {'n':>4}{'binding (eV)':>16}{'ionizing field':>18}")
    for n in (1, 5, 10, 30, 50):
        b = hydrogen_binding_energy(n)
        F = field_ionization_threshold(b)
        fs = f"{F:.2e} V/m"
        print(f"  {n:>4}{b/EV:>14.4f}{fs:>18}")

    print("\n  Hydrogen's degenerate levels give the LINEAR Stark effect (a permanent dipole);")
    print("  most atoms shift quadratically as -1/2 alpha E^2. And because binding ~ 1/n^2, a")
    print("  Rydberg atom ionizes in a field ten billion times weaker than the ground state --")
    print("  which is how Rydberg-atom detectors sense tiny fields and single microwave photons.")

    _svg(os.path.join(outdir, "stark.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'stark.svg')}")


def _svg(path, size=720, pad=76):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The Stark effect</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'hydrogen n=4 levels fanning out linearly with field (top); ionization field vs n (bottom)</text>',
    ]

    x0, x1 = pad, size - pad
    mid = size * 0.54

    # --- top: linear Stark fan for n=4 (7 components) ---
    ty0, ty1 = mid - 30, pad + 44
    Emax = 1e7
    n = 4
    comps_max = linear_stark_pattern(n, Emax)
    span = max(abs(c) for c in comps_max) * 1.15
    def FX(E):
        return x0 + E / Emax * (x1 - x0)
    cy = (ty0 + ty1) / 2
    def SY(dE):
        return cy - dE / span * (ty0 - ty1) / 2
    parts.append(f'<line x1="{x0}" y1="{cy:.1f}" x2="{x1}" y2="{cy:.1f}" stroke="#21262d" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x0}" y2="{ty1}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x1}" y2="{ty0}" stroke="#8b949e" stroke-width="1.4"/>')
    ncomp = len(linear_stark_pattern(n, Emax))
    cols = ["#4dabf7", "#06d6a0", "#ff922b", "#ffd43b", "#ff6b6b", "#8338ec", "#b197fc"]
    for idx in range(ncomp):
        pts = []
        for i in range(41):
            E = Emax * i / 40
            pts.append(f"{FX(E):.1f},{SY(linear_stark_pattern(n, E)[idx]):.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{cols[idx%len(cols)]}" stroke-width="2"/>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{ty0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">electric field -> (n=4 splits into 7 equally spaced lines)</text>')
    parts.append(f'<text x="{x0-24:.1f}" y="{cy:.1f}" fill="#8b949e" font-size="10" '
                 f'transform="rotate(-90 {x0-24:.1f} {cy:.1f})" text-anchor="middle">energy shift</text>')

    # --- bottom: ionization field vs n (log y) ---
    by0, by1 = size - pad, mid + 40
    ns = list(range(1, 61))
    Fs = [field_ionization_threshold(hydrogen_binding_energy(nn)) for nn in ns]
    ly0, ly1 = math.log10(min(Fs)), math.log10(max(Fs))
    def NX(nn):
        return x0 + (nn - 1) / 59 * (x1 - x0)
    def FY(F):
        return by0 - (math.log10(F) - ly0) / (ly1 - ly0) * (by0 - by1)
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x1}" y2="{by0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x0}" y2="{by1}" stroke="#8b949e" stroke-width="1.4"/>')
    for e in range(int(math.floor(ly0)), int(math.ceil(ly1)) + 1, 2):
        gy = FY(10.0 ** e)
        if by1 <= gy <= by0:
            parts.append(f'<text x="{x0-6:.1f}" y="{gy+3:.1f}" fill="#8b949e" font-size="9" '
                         f'text-anchor="end">10^{e}</text>')
    for nn in (1, 10, 20, 30, 40, 50, 60):
        parts.append(f'<text x="{NX(nn):.1f}" y="{by0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{nn}</text>')
    poly = " ".join(f"{NX(ns[i]):.1f},{FY(Fs[i]):.1f}" for i in range(len(ns)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ff6b6b" stroke-width="2.6"/>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{by0+28:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">principal quantum number n (ionizing field, V/m, log scale)</text>')
    parts.append(f'<text x="{NX(20):.1f}" y="{FY(Fs[5]):.1f}" fill="#ff6b6b" font-size="10">'
                 f'F_ion ~ 1/n^4: Rydberg atoms ionize in tiny fields</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
