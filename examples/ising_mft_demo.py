"""Demo: mean-field Ising ferromagnetism -- order emerging at the Curie point.

Prints the spontaneous magnetization as temperature crosses the Curie point and the diverging
susceptibility above it, then draws the magnetization curve m(T) collapsing to zero at T_c
alongside the Curie-Weiss susceptibility blowing up.

    python examples/ising_mft_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ising_mft import (curie_temperature, spontaneous_magnetization,  # noqa: E402
                       is_ferromagnetic, curie_weiss_susceptibility)


J, Z = 1.0, 6   # coupling, coordination (simple cubic)
TC = curie_temperature(J, Z)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Mean-field Ising: m = tanh((z J m + B)/T), T_c = z J = %g\n" % TC)
    print(f"  {'T / T_c':>10}{'magnetization':>16}{'phase':>16}")
    for frac in (0.2, 0.5, 0.8, 0.95, 1.0, 1.2):
        T = frac * TC
        m = spontaneous_magnetization(T, J, Z)
        phase = "ferromagnet" if is_ferromagnetic(T, J, Z) else "paramagnet"
        print(f"  {frac:>10.2f}{m:>16.3f}{phase:>16}")

    print("\n  Curie-Weiss susceptibility above T_c (chi = C/(T - T_c)):")
    for frac in (1.05, 1.2, 1.5, 2.0):
        print(f"    T = {frac:.2f} T_c  ->  chi = {curie_weiss_susceptibility(frac*TC, TC):.3f}")

    print("\n  Below T_c countless spins tip collectively into one direction with no applied")
    print("  field -- spontaneous symmetry breaking. The magnetization vanishes as (1-T/Tc)^1/2")
    print("  (mean-field beta=1/2) and the susceptibility diverges at T_c: a phase transition.")

    _svg(os.path.join(outdir, "ising_mft.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'ising_mft.svg')}")


def _svg(path, size=720, pad=76):
    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 44
    t_max = 2.0   # in units of T_c

    def X(frac):
        return x0 + frac / t_max * (x1 - x0)

    def MY(m):
        return y0 - m * (y0 - y1)          # m in [0,1]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Mean-field ferromagnetism</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'magnetization collapses to zero at the Curie point; susceptibility diverges there</text>',
    ]

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.5"/>')
    for m in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{x0-6:.1f}" y="{MY(m)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{m:.1f}</text>')

    # T_c marker + phase shading
    parts.append(f'<rect x="{x0:.1f}" y="{y1:.1f}" width="{X(1.0)-x0:.1f}" height="{y0-y1:.1f}" '
                 f'fill="#06d6a0" opacity="0.05"/>')
    parts.append(f'<line x1="{X(1.0):.1f}" y1="{y0:.1f}" x2="{X(1.0):.1f}" y2="{y1:.1f}" '
                 f'stroke="#ff6b6b" stroke-width="1.2" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{X(1.0):.1f}" y="{y1-6:.1f}" fill="#ff6b6b" font-size="11" '
                 f'text-anchor="middle">T_c</text>')
    parts.append(f'<text x="{X(0.4):.1f}" y="{y1+14:.1f}" fill="#06d6a0" font-size="10" '
                 f'text-anchor="middle">ferromagnet (m != 0)</text>')
    parts.append(f'<text x="{X(1.5):.1f}" y="{y1+14:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">paramagnet (m = 0)</text>')

    # magnetization curve
    pts = []
    n = 200
    for i in range(1, n + 1):
        frac = t_max * i / n
        pts.append(f"{X(frac):.1f},{MY(spontaneous_magnetization(frac * TC, J, Z)):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="2.8"/>')
    parts.append(f'<text x="{X(0.25):.1f}" y="{MY(0.9):.1f}" fill="#4dabf7" font-size="11">'
                 f'magnetization m(T)</text>')

    # susceptibility curve (scaled to fit), diverging at T_c from above
    cmax = curie_weiss_susceptibility(1.05 * TC, TC)
    spts = []
    frac = 1.03
    while frac <= t_max:
        chi = curie_weiss_susceptibility(frac * TC, TC)
        spts.append(f"{X(frac):.1f},{MY(min(1.0, chi / cmax)):.1f}")
        frac += 0.01
    parts.append(f'<polyline points="{" ".join(spts)}" fill="none" stroke="#ff922b" stroke-width="2.2"/>')
    parts.append(f'<text x="{X(1.35):.1f}" y="{MY(0.55):.1f}" fill="#ff922b" font-size="11">'
                 f'susceptibility chi ~ 1/(T-T_c)</text>')

    for frac in (0.5, 1.0, 1.5, 2.0):
        parts.append(f'<text x="{X(frac):.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{frac:.1f} T_c</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+32:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">temperature</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
