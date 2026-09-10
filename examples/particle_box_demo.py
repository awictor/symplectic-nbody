"""Demo: the particle in a box.

Prints the quantized levels of an electron in a nanometre box and the size-tunable
quantum-dot emission colours, then draws the first few energy levels with their
wavefunctions superimposed inside the well.

    python examples/particle_box_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from particle_box import (energy_level, ground_state_energy, level_spacing,  # noqa: E402
                          transition_wavelength, wavefunction, box_width_for_gap,
                          EV, M_E)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    L = 1e-9
    print(f"Particle in a box: E_n = n^2 h^2 / (8 m L^2)  (electron, L = 1 nm)\n")
    print(f"  {'n':>4}{'E_n (eV)':>11}{'gap to n-1 (eV)':>18}")
    print("  " + "-" * 34)
    for n in range(1, 6):
        gap = (energy_level(n, L) - energy_level(n - 1, L)) / EV if n > 1 else 0.0
        print(f"  {n:>4}{energy_level(n, L)/EV:>11.3f}{gap:>18.3f}")

    print("\n  quantum-dot n=1->2 emission (smaller box = bluer):")
    for L_nm in (5.0, 3.0, 2.0, 1.0):
        lam = transition_wavelength(1, 2, L_nm * 1e-9) * 1e9
        print(f"    {L_nm:.0f} nm dot  ->  {lam:.0f} nm")

    print("\n  Only whole numbers of half-wavelengths fit between the walls, so energy")
    print("  comes in n^2 rungs with a nonzero ground state (confinement zero-point")
    print("  energy). Because every level scales as 1/L^2, shrinking a quantum dot")
    print("  widens the gaps and shifts its glow toward the blue -- size-tunable colour,")
    print("  used in displays and biological markers.")

    _svg(os.path.join(outdir, "particle_box.svg"), L)
    print(f"\n  wrote {os.path.join(outdir, 'particle_box.svg')}")


def _svg(path, L, size=720, pad=72, n_show=4):
    E = [energy_level(n, L) / EV for n in range(1, n_show + 1)]
    emax = E[-1] * 1.15

    x0, x1 = pad + 30, size - pad - 30

    def ex(x):
        return x0 + x / L * (x1 - x0)

    def ey(e):
        return size - pad - e / emax * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # well walls
    parts.append(f'<line x1="{x0:.1f}" y1="{pad}" x2="{x0:.1f}" y2="{size-pad}" stroke="#30363d" stroke-width="2"/>')
    parts.append(f'<line x1="{x1:.1f}" y1="{pad}" x2="{x1:.1f}" y2="{size-pad}" stroke="#30363d" stroke-width="2"/>')
    parts.append(f'<line x1="{x0:.1f}" y1="{size-pad:.1f}" x2="{x1:.1f}" y2="{size-pad:.1f}" stroke="#30363d" stroke-width="2"/>')

    cols = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff6b6b"]
    N = 200
    for i, n in enumerate(range(1, n_show + 1)):
        y_level = ey(E[i])
        col = cols[i % len(cols)]
        # level line
        parts.append(f'<line x1="{x0:.1f}" y1="{y_level:.1f}" x2="{x1:.1f}" y2="{y_level:.1f}" '
                     f'stroke="{col}" stroke-width="1" stroke-dasharray="3 3" opacity="0.5"/>')
        parts.append(f'<text x="{x0-8:.1f}" y="{y_level+4:.1f}" fill="{col}" '
                     f'font-size="10" text-anchor="end">n={n}</text>')
        parts.append(f'<text x="{x1+6:.1f}" y="{y_level+4:.1f}" fill="{col}" '
                     f'font-size="9">{E[i]:.2f} eV</text>')
        # wavefunction riding on the level line
        amp = (size - 2 * pad) / n_show * 0.42
        pts = []
        for j in range(N + 1):
            x = L * j / N
            psi = math.sin(n * math.pi * x / L)
            pts.append(f"{ex(x):.1f},{y_level - psi*amp:.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2"/>')

    parts.append(f'<text x="20" y="34" fill="#e6edf3" font-size="18">'
                 f'Particle in a box: levels and wavefunctions</text>')
    parts.append(f'<text x="20" y="52" fill="#8b949e" font-size="12">'
                 f'E_n ~ n^2; psi_n has n-1 nodes; nonzero ground state</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{size-pad+18:.1f}" fill="#8b949e" '
                 f'font-size="10" text-anchor="middle">position across the well (width L)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
