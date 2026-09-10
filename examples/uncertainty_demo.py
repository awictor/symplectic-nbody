"""Demo: the Heisenberg uncertainty principle.

Prints the zero-point confinement energy for particles squeezed into boxes from atomic
to nuclear size, then draws confinement energy vs box size for an electron and a
nucleon, with the atomic and nuclear scales marked -- the eV and MeV worlds emerging
from dx dp >= hbar/2.

    python examples/uncertainty_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uncertainty import (min_momentum_spread, confinement_energy,  # noqa: E402
                         hydrogen_ground_state_estimate, natural_linewidth_hz,
                         M_E, M_P, EV, MEV, FM)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Heisenberg: dx dp >= hbar/2 -> irreducible confinement energy\n")
    print(f"  {'confinement':>20}{'size':>10}{'electron E':>14}{'nucleon E':>14}")
    print("  " + "-" * 58)
    boxes = [
        ("molecule", 1e-9),
        ("atom", 1e-10),
        ("inner atom", 1e-11),
        ("atomic nucleus", 5e-15),
        ("nucleon core", 1e-15),
    ]
    for name, dx in boxes:
        Ee = confinement_energy(dx, M_E)
        Ep = confinement_energy(dx, M_P)
        estr = f"{Ee/EV:.2g} eV" if Ee < 1e3 * EV else f"{Ee/MEV:.2g} MeV"
        pstr = f"{Ep/EV:.2g} eV" if Ep < 1e3 * EV else f"{Ep/MEV:.2g} MeV"
        sstr = f"{dx*1e9:.2g} nm" if dx >= 1e-12 else f"{dx/FM:.2g} fm"
        print(f"  {name:>20}{sstr:>10}{estr:>14}{pstr:>14}")

    print(f"\n  minimizing confinement + Coulomb energy gives hydrogen's binding:")
    print(f"    {hydrogen_ground_state_estimate()/EV:.2f} eV -- the atomic scale from uncertainty alone\n")
    print("  Squeezing a particle into a smaller box forces a larger momentum spread")
    print("  and thus more kinetic energy (E ~ 1/dx^2). That is why electrons don't")
    print("  fall into the nucleus, why atoms are ~0.1 nm (eV energies), and why")
    print("  nucleons confined to femtometres carry MeV -- the energy scale of nuclei.")

    _svg(os.path.join(outdir, "uncertainty.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'uncertainty.svg')}")


def _svg(path, size=720, pad=72):
    dxs = [10 ** (-15 + 0.08 * i) for i in range(0, 91)]   # 1 fm .. ~1e-7 m
    Ee = [confinement_energy(dx, M_E) / EV for dx in dxs]
    Ep = [confinement_energy(dx, M_P) / EV for dx in dxs]
    lx = [math.log10(dx) for dx in dxs]
    lee = [math.log10(e) for e in Ee]
    lep = [math.log10(e) for e in Ep]
    xmin, xmax = lx[0], lx[-1]
    ymin, ymax = min(lep), max(lee)

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    # scale markers: atomic (0.1 nm), nuclear (few fm)
    for dx, label, col in [(1e-10, "atom (0.1 nm)", "#06d6a0"),
                           (5e-15, "nucleus (5 fm)", "#ff922b")]:
        x = sx(math.log10(dx))
        parts.append(f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{size-pad}" '
                     f'stroke="{col}" stroke-width="1" stroke-dasharray="4 4"/>')
        parts.append(f'<text x="{x+4:.1f}" y="{pad+40:.1f}" fill="{col}" '
                     f'font-size="10" transform="rotate(90 {x+4:.1f} {pad+40:.1f})">{label}</text>')
    # eV and MeV horizontal guides
    for E_ref, label, col in [(1.0, "1 eV", "#8b949e"), (1e6, "1 MeV", "#8b949e")]:
        if ymin <= math.log10(E_ref) <= ymax:
            y = sy(math.log10(E_ref))
            parts.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{size-pad}" y2="{y:.1f}" '
                         f'stroke="{col}" stroke-width="0.8" stroke-dasharray="2 5" opacity="0.6"/>')
            parts.append(f'<text x="{size-pad-4:.1f}" y="{y-4:.1f}" fill="{col}" '
                         f'font-size="10" text-anchor="end">{label}</text>')

    pe = " ".join(f"{sx(lx[i]):.1f},{sy(lee[i]):.1f}" for i in range(len(dxs)))
    pp = " ".join(f"{sx(lx[i]):.1f},{sy(lep[i]):.1f}" for i in range(len(dxs)))
    parts.append(f'<polyline points="{pe}" fill="none" stroke="#4dabf7" stroke-width="2.4"/>')
    parts.append(f'<polyline points="{pp}" fill="none" stroke="#ff6b6b" stroke-width="2.4"/>')

    parts.append(f'<text x="{pad+10}" y="{pad+22}" fill="#4dabf7" font-size="12">electron</text>')
    parts.append(f'<text x="{pad+10}" y="{pad+38}" fill="#ff6b6b" font-size="12">nucleon (proton)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Confinement (zero-point) energy vs box size</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'E ~ hbar^2 / (m dx^2): tighter box, higher irreducible energy</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 confinement size dx (m) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 confinement energy (eV)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
