"""Demo: the Sackur-Tetrode absolute entropy of an ideal gas.

Prints the predicted standard molar entropy of the noble gases against measured
values, then draws molar entropy vs temperature for several gases -- the log-linear
rise that quantum state-counting predicts from first principles.

    python examples/sackur_tetrode_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sackur_tetrode import (molar_entropy, thermal_wavelength,  # noqa: E402
                            entropy_per_particle, AMU)

STP_T, STP_P = 298.15, 101325.0
# (name, mass amu, measured standard molar entropy J/mol/K)
GASES = [
    ("He", 4.0026, 126.2),
    ("Ne", 20.180, 146.3),
    ("Ar", 39.948, 154.8),
    ("Kr", 83.798, 164.1),
    ("Xe", 131.29, 169.7),
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Sackur-Tetrode: absolute entropy from counting quantum microstates\n")
    print(f"  standard molar entropy at STP (298.15 K, 1 atm):\n")
    print(f"  {'gas':>6}{'mass (amu)':>12}{'S predicted':>14}{'S measured':>13}"
          f"{'error':>9}")
    print("  " + "-" * 54)
    for name, amu, meas in GASES:
        S = molar_entropy(STP_T, STP_P, amu * AMU)
        err = (S - meas) / meas * 100
        print(f"  {name:>6}{amu:>12.3f}{S:>14.1f}{meas:>13.1f}{err:>8.2f}%")

    print("\n  Predicted from nothing but atomic mass, T and P -- and matching the")
    print("  calorimetric values to a fraction of a percent. The quantum of action h")
    print("  appears explicitly (a phase-space cell is h^3), and the 1/N! for")
    print("  indistinguishable atoms is what makes entropy extensive and resolves the")
    print("  Gibbs paradox. Entropy really is the logarithm of countable microstates.")

    _svg(os.path.join(outdir, "sackur_tetrode.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'sackur_tetrode.svg')}")


def _svg(path, size=720, pad=72):
    Ts = [50.0 + 20.0 * i for i in range(0, 71)]   # 50 .. ~1450 K
    curves = [("He", 4.0026, "#4dabf7"), ("Ar", 39.948, "#ffd43b"),
              ("Xe", 131.29, "#ff6b6b")]
    data = []
    for name, amu, col in curves:
        ys = [molar_entropy(T, STP_P, amu * AMU) for T in Ts]
        data.append((name, col, ys))

    allv = [v for _, _, ys in data for v in ys]
    xmin, xmax = Ts[0], Ts[-1]
    ymin, ymax = min(allv), max(allv)

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
    for i, (name, col, ys) in enumerate(data):
        poly = " ".join(f"{sx(Ts[j]):.1f},{sy(ys[j]):.1f}" for j in range(len(Ts)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        parts.append(f'<text x="{pad+10}" y="{ytop + i*16}" fill="{col}" '
                     f'font-size="12">{name}</text>')

    # STP measured points
    for name, amu, meas in GASES:
        col = {"He": "#4dabf7", "Ar": "#ffd43b", "Xe": "#ff6b6b"}.get(name)
        if col and ymin <= meas <= ymax:
            parts.append(f'<circle cx="{sx(STP_T):.1f}" cy="{sy(meas):.1f}" r="4" '
                         f'fill="none" stroke="{col}" stroke-width="1.6"/>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Absolute entropy vs temperature (Sackur-Tetrode)</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'rings mark measured STP values; heavier gas = more entropy</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">temperature (K) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'molar entropy (J/mol/K)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
