"""Demo: Fermi degeneracy pressure -- the quantum support of dead stars.

Shows the degeneracy pressure across densities, the softening of the exponent
from 5/3 (non-relativistic) to 4/3 (relativistic) that creates the Chandrasekhar
mass, and renders both pressure laws vs density to a log-log SVG.

    python examples/degeneracy_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from degeneracy import (fermi_momentum, pressure_nonrel,  # noqa: E402
                        pressure_relativistic, is_relativistic,
                        transition_density, M_E, C)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Fermi degeneracy pressure: the quantum floor under dead stars\n")
    print(f"  relativistic transition density: {transition_density():.2e} /m^3\n")
    print(f"  {'n (/m^3)':>12}{'regime':>18}{'P (Pa)':>14}")
    print("  " + "-" * 46)
    for nexp in (29, 32, 35, 36, 38):
        n = 10.0 ** nexp
        rel = is_relativistic(n)
        P = pressure_relativistic(n) if rel else pressure_nonrel(n)
        print(f"  {10.0**nexp:>12.0e}{'relativistic' if rel else 'non-rel':>18}{P:>14.2e}")
    print("\n  Below the transition the pressure goes as n^5/3; above it, as n^4/3.")
    print("  That softer exponent is exactly why gravity eventually wins in a")
    print("  massive white dwarf -- the origin of the Chandrasekhar mass.")

    _svg(os.path.join(outdir, "degeneracy.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'degeneracy.svg')}")


def _svg(path, size=720, pad=64):
    ns = [10 ** (30 + 0.08 * i) for i in range(0, 101)]  # 1e30 .. 1e38
    Pnr = [pressure_nonrel(n) for n in ns]
    Pr = [pressure_relativistic(n) for n in ns]
    ln = [math.log10(n) for n in ns]
    lnr = [math.log10(p) for p in Pnr]
    lr = [math.log10(p) for p in Pr]
    nmin, nmax = ln[0], ln[-1]
    ymin = min(min(lnr), min(lr))
    ymax = max(max(lnr), max(lr))

    def sx(x):
        return pad + (x - nmin) / (nmax - nmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    # transition line
    lt = math.log10(transition_density())
    parts.append(f'<line x1="{sx(lt):.1f}" y1="{pad}" x2="{sx(lt):.1f}" y2="{size-pad}" '
                 f'stroke="#e9c46a" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{sx(lt)+6:.1f}" y="{pad+16}" fill="#e9c46a" font-size="12">'
                 f'p_F = m_e c</text>')
    nrp = " ".join(f"{sx(ln[i]):.1f},{sy(lnr[i]):.1f}" for i in range(len(ns)))
    rp = " ".join(f"{sx(ln[i]):.1f},{sy(lr[i]):.1f}" for i in range(len(ns)))
    parts.append(f'<polyline points="{nrp}" fill="none" stroke="#4cc9f0" stroke-width="2"/>')
    parts.append(f'<polyline points="{rp}" fill="none" stroke="#ff006e" stroke-width="2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Degeneracy pressure vs density (log-log)</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+8}" fill="#4cc9f0" font-size="12">'
                 f'non-rel P ~ n^5/3 (steeper)</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+26}" fill="#ff006e" font-size="12">'
                 f'relativistic P ~ n^4/3 (softer -> Chandrasekhar)</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 n (/m^3) -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 P (Pa)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
