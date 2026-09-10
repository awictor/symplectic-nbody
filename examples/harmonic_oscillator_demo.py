"""Demo: the quantum harmonic oscillator.

Prints the vibrational quantum and infrared wavelength for a few diatomic molecules,
then draws the equally-spaced energy levels sitting inside the parabolic potential well,
with the nonzero zero-point ground state marked.

    python examples/harmonic_oscillator_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harmonic_oscillator import (angular_frequency, energy_level,  # noqa: E402
                                 zero_point_energy, level_spacing,
                                 transition_wavelength, turning_point, EV, AMU)


def _mu(m1, m2):
    return m1 * m2 / (m1 + m2) * AMU


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Quantum harmonic oscillator: E_n = (n + 1/2) hbar omega\n")
    print(f"  {'molecule':>10}{'k (N/m)':>10}{'hbar omega (eV)':>18}"
          f"{'IR wavelength':>16}")
    print("  " + "-" * 54)
    mols = [
        ("H2", 570.0, _mu(1, 1)),
        ("CO", 1902.0, _mu(12, 16)),
        ("N2", 2294.0, _mu(14, 14)),
        ("HCl", 516.0, _mu(1, 35)),
    ]
    for name, k, mu in mols:
        dE = level_spacing(k, mu) / EV
        lam = transition_wavelength(k, mu) * 1e6
        print(f"  {name:>10}{k:>10.0f}{dE:>18.4f}{lam:>13.2f} um")

    print("\n  Unlike the box (n^2) or atom (-1/n^2), the oscillator's levels are EVENLY")
    print("  spaced by hbar omega, so a molecule absorbs a single sharp infrared line")
    print("  per vibrational quantum. The ground state is not zero -- the zero-point")
    print("  energy (1/2) hbar omega is forced by the uncertainty principle, and it is")
    print("  real: it keeps helium liquid at absolute zero and shifts bond energies.")

    _svg(os.path.join(outdir, "harmonic_oscillator.svg"), 1902.0, _mu(12, 16))
    print(f"\n  wrote {os.path.join(outdir, 'harmonic_oscillator.svg')}")


def _svg(path, k, mu, size=680, n_show=6):
    pad = 72
    E = [energy_level(n, k, mu) / EV for n in range(n_show)]
    emax = E[-1] * 1.25
    cx = size / 2.0

    def ey(e):
        return size - pad - e / emax * (size - 2 * pad)

    # parabola V = 1/2 k x^2 in eV; map x so the top level's turning point spans width
    x_max = turning_point(n_show - 1, k, mu) * 1.15
    half_w = (size / 2.0 - pad)

    def px(x):
        return cx + x / x_max * half_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # potential well parabola
    pts = []
    N = 120
    for i in range(N + 1):
        x = -x_max + 2 * x_max * i / N
        V = 0.5 * k * x * x / EV
        if V <= emax:
            pts.append(f"{px(x):.1f},{ey(V):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#8b949e" stroke-width="2"/>')

    cols = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#ff6b6b", "#b197fc"]
    for n in range(n_show):
        e = E[n]
        xt = turning_point(n, k, mu)
        y = ey(e)
        col = cols[n % len(cols)]
        parts.append(f'<line x1="{px(-xt):.1f}" y1="{y:.1f}" x2="{px(xt):.1f}" y2="{y:.1f}" '
                     f'stroke="{col}" stroke-width="2"/>')
        label = "n=0 (zero-point)" if n == 0 else f"n={n}"
        parts.append(f'<text x="{px(xt)+6:.1f}" y="{y+4:.1f}" fill="{col}" '
                     f'font-size="10">{label}</text>')

    parts.append(f'<text x="20" y="34" fill="#e6edf3" font-size="18">'
                 f'Harmonic oscillator: evenly-spaced levels (CO)</text>')
    parts.append(f'<text x="20" y="52" fill="#8b949e" font-size="12">'
                 f'gap = hbar omega = {level_spacing(k, mu)/EV:.3f} eV; ground state at half that</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
