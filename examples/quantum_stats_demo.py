"""Demo: Fermi-Dirac, Bose-Einstein, and Maxwell-Boltzmann statistics.

Prints the occupation of the three distributions across energies, then draws them
together vs (E-mu)/kT -- the fermion step capped at 1, the boson divergence near mu, and
their merge into the classical exponential far above mu.

    python examples/quantum_stats_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantum_stats import (fermi_dirac, bose_einstein, maxwell_boltzmann,  # noqa: E402
                           fermi_step_width, K_B, EV)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    mu = 0.0
    T = 300.0
    kT = K_B * T
    print("Quantum statistics: occupation <n> vs (E-mu)/kT\n")
    print(f"  {'(E-mu)/kT':>12}{'Fermi-Dirac':>14}{'Bose-Einstein':>16}"
          f"{'Maxwell-Boltz':>16}")
    print("  " + "-" * 58)
    for x in (-4, -1, 0.5, 1, 2, 4, 8):
        E = mu + x * kT
        fd = fermi_dirac(E, mu, T)
        be = bose_einstein(E, mu, T) if x > 0 else float("inf")
        mb = maxwell_boltzmann(E, mu, T)
        bestr = f"{be:.3f}" if be != float("inf") else "inf"
        print(f"  {x:>12.1f}{fd:>14.4f}{bestr:>16}{mb:>16.4f}")

    print("\n  Fermions can never exceed one per state (Pauli), so at low temperature")
    print("  they stack into a sharp step up to the Fermi level -- electron degeneracy,")
    print("  white-dwarf pressure. Bosons pile up without limit as E -> mu, condensing")
    print("  into the ground state below a critical temperature (BEC, superfluid He).")
    print("  Far above mu both fade into the classical Maxwell-Boltzmann exponential.")

    _svg(os.path.join(outdir, "quantum_stats.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'quantum_stats.svg')}")


def _svg(path, size=720, pad=72):
    mu, T = 0.0, 300.0
    kT = K_B * T
    xs = [-5 + 0.1 * i for i in range(0, 131)]   # (E-mu)/kT from -5 to +8
    fd = [fermi_dirac(mu + x * kT, mu, T) for x in xs]
    be = [min(bose_einstein(mu + x * kT, mu, T), 3.0) if x > 0.02 else None for x in xs]
    mb = [min(maxwell_boltzmann(mu + x * kT, mu, T), 3.0) for x in xs]
    xmin, xmax = xs[0], xs[-1]
    ymin, ymax = 0.0, 3.0

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
    # E = mu vertical line and <n>=1 line
    xm = sx(0.0)
    parts.append(f'<line x1="{xm:.1f}" y1="{pad}" x2="{xm:.1f}" y2="{size-pad}" '
                 f'stroke="#30363d" stroke-width="0.8" stroke-dasharray="2 4"/>')
    parts.append(f'<text x="{xm+4:.1f}" y="{size-pad-6:.1f}" fill="#8b949e" font-size="10">E = mu</text>')
    y1 = sy(1.0)
    parts.append(f'<line x1="{pad}" y1="{y1:.1f}" x2="{size-pad}" y2="{y1:.1f}" '
                 f'stroke="#30363d" stroke-width="0.8" stroke-dasharray="2 4"/>')

    def poly(vals, col):
        pts = " ".join(f"{sx(xs[i]):.1f},{sy(vals[i]):.1f}"
                       for i in range(len(xs)) if vals[i] is not None)
        return f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>'
    parts.append(poly(fd, "#4dabf7"))
    parts.append(poly(be, "#ff6b6b"))
    parts.append(poly(mb, "#ffd43b"))

    parts.append(f'<text x="{pad+10}" y="{pad+22}" fill="#4dabf7" font-size="12">Fermi-Dirac (fermions)</text>')
    parts.append(f'<text x="{pad+10}" y="{pad+38}" fill="#ff6b6b" font-size="12">Bose-Einstein (bosons)</text>')
    parts.append(f'<text x="{pad+10}" y="{pad+54}" fill="#ffd43b" font-size="12">Maxwell-Boltzmann (classical)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Occupation of the three statistics</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">(E - mu) / kT -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'average occupation &lt;n&gt;</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
