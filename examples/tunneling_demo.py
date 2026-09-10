"""Demo: quantum tunneling through a barrier.

Prints transmission probabilities for an electron through barriers of varying width and
height, then draws the transmission vs barrier width -- the exponential decay that makes
tunneling both vanishingly rare for thick barriers and exquisitely sensitive for thin ones.

    python examples/tunneling_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tunneling import (decay_constant, transmission_thick, transmission_exact,  # noqa: E402
                       stm_current_ratio, EV)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Quantum tunneling: T ~ exp(-2 kappa L), kappa = sqrt(2m(V-E))/hbar\n")
    print(f"  {'V - E (eV)':>12}{'width (nm)':>12}{'transmission':>16}")
    print("  " + "-" * 40)
    for dV in (1.0, 4.0):
        for L_nm in (0.2, 0.5, 1.0):
            T = transmission_thick(dV * EV, L_nm * 1e-9)
            print(f"  {dV:>12.0f}{L_nm:>12.1f}{T:>16.2e}")

    print("\n  STM tip-surface gap sensitivity (4 eV work function):")
    for dnm in (0.1, 0.2, 0.3):
        r = stm_current_ratio(0.0, dnm * 1e-9)
        print(f"    +{dnm:.1f} nm gap  ->  current x {r:.2e}  (1/{1/r:.0f})")

    print("\n  Transmission plunges exponentially with width, so a barrier a nanometre")
    print("  thick is essentially opaque -- yet shave off an Angstrom and the current")
    print("  jumps ~8x. That razor sensitivity is how a scanning tunneling microscope")
    print("  feels individual atoms, and the same barrier penetration drives alpha")
    print("  decay (tunneling out) and stellar fusion (tunneling in).")

    _svg(os.path.join(outdir, "tunneling.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'tunneling.svg')}")


def _svg(path, size=720, pad=72):
    Ls = [0.05e-9 + 0.02e-9 * i for i in range(0, 75)]   # 0.05 .. ~1.5 nm
    curves = [(1.0, "#4dabf7", "V-E = 1 eV"), (4.0, "#ffd43b", "V-E = 4 eV"),
              (10.0, "#ff6b6b", "V-E = 10 eV")]
    data = []
    for dV, col, label in curves:
        ys = [math.log10(max(transmission_thick(dV * EV, L), 1e-300)) for L in Ls]
        data.append((ys, col, label))

    lx = [L * 1e9 for L in Ls]   # nm
    xmin, xmax = lx[0], lx[-1]
    all_y = [y for ys, _, _ in data for y in ys]
    ymin, ymax = max(min(all_y), -40.0), 0.0

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
    ytop = pad + 22
    for i, (ys, col, label) in enumerate(data):
        pts = " ".join(f"{sx(lx[j]):.1f},{sy(max(ys[j], ymin)):.1f}" for j in range(len(Ls)))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        parts.append(f'<text x="{size-pad-120}" y="{ytop + i*16}" fill="{col}" '
                     f'font-size="12">{label}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Tunneling transmission vs barrier width (electron)</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'straight lines on a log plot: T falls exponentially with width</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">barrier width (nm) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 transmission probability</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
