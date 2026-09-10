"""Demo: the semi-empirical mass formula and the binding-energy curve.

Prints the binding energy per nucleon for landmark nuclei, then draws the famous curve
of binding energy per nucleon vs mass number, peaking near iron -- the reason fusion
powers stars up to iron and fission powers reactors beyond it.

    python examples/mass_formula_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mass_formula import (binding_energy, binding_per_nucleon,  # noqa: E402
                          most_stable_Z)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Semi-empirical mass formula: B/A peaks at iron (~8.8 MeV/nucleon)\n")
    print(f"  {'nucleus':>12}{'Z':>5}{'A':>5}{'B/A (MeV)':>12}")
    print("  " + "-" * 34)
    nuclei = [
        ("He-4", 2, 4), ("C-12", 6, 12), ("O-16", 8, 16),
        ("Fe-56", 26, 56), ("Ni-62", 28, 62), ("Kr-84", 36, 84),
        ("Sn-120", 50, 120), ("Pb-208", 82, 208), ("U-238", 92, 238),
    ]
    for name, Z, A in nuclei:
        print(f"  {name:>12}{Z:>5}{A:>5}{binding_per_nucleon(Z, A):>12.3f}")

    best = max(range(20, 120), key=lambda A: binding_per_nucleon(most_stable_Z(A), A))
    print(f"\n  peak of the curve: A = {best}, "
          f"{binding_per_nucleon(most_stable_Z(best), best):.2f} MeV/nucleon\n")
    print("  Light nuclei fuse toward the peak, releasing energy; heavy nuclei fission")
    print("  toward it, also releasing energy. Iron-56 sits at the summit -- the ash of")
    print("  stellar fusion and the end of the line for energy release, which is why a")
    print("  massive star's iron core cannot burn and collapses into a supernova.")

    _svg(os.path.join(outdir, "mass_formula.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'mass_formula.svg')}")


def _svg(path, size=720, pad=72):
    As = list(range(4, 250))
    ba = [binding_per_nucleon(most_stable_Z(A), A) for A in As]
    xmin, xmax = As[0], As[-1]
    ymin, ymax = 0.0, 9.5

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
    # peak marker (iron)
    best = max(As, key=lambda A: binding_per_nucleon(most_stable_Z(A), A))
    xf = sx(best)
    parts.append(f'<line x1="{xf:.1f}" y1="{pad}" x2="{xf:.1f}" y2="{size-pad}" '
                 f'stroke="#ffd43b" stroke-width="1" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{xf+5:.1f}" y="{pad+40:.1f}" fill="#ffd43b" '
                 f'font-size="11">iron peak (A~{best})</text>')
    # fusion / fission arrows
    parts.append(f'<text x="{sx(20):.1f}" y="{sy(3.0):.1f}" fill="#06d6a0" '
                 f'font-size="12">fusion -&gt;</text>')
    parts.append(f'<text x="{sx(200):.1f}" y="{sy(6.2):.1f}" fill="#ff6b6b" '
                 f'font-size="12" text-anchor="end">&lt;- fission</text>')

    poly = " ".join(f"{sx(As[i]):.1f},{sy(ba[i]):.1f}" for i in range(len(As)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')

    for name, A, col in [("He-4", 4, "#8b949e"), ("Fe-56", 56, "#ffd43b"),
                         ("U-238", 238, "#ff6b6b")]:
        Z = most_stable_Z(A)
        px = sx(A)
        py = sy(binding_per_nucleon(Z, A))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{col}"/>')
        parts.append(f'<text x="{px+6:.1f}" y="{py+10:.1f}" fill="{col}" '
                     f'font-size="9">{name}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Binding energy per nucleon vs mass number</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'the curve of nuclear stability: iron at the summit</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">mass number A -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'B/A (MeV per nucleon)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
