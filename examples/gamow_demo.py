"""Demo: the Gamow peak of thermonuclear fusion.

Prints the Gamow peak energy for several reactions and temperatures, then draws the
two competing factors -- the falling Maxwell-Boltzmann tail and the rising tunnelling
probability -- and their product, the sharply-peaked Gamow window where fusion happens.

    python examples/gamow_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gamow import (reduced_mass, gamow_energy, gamow_peak_energy,  # noqa: E402
                   gamow_peak_width, integrand, peak_energy_kev, M_P, K_B, KEV)

MU_PP = reduced_mass(M_P, M_P)
T_SUN = 1.5e7


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Gamow peak: Boltzmann tail x quantum tunnelling = fusion window\n")
    print(f"  solar core kT = {K_B*T_SUN/KEV:.2f} keV; barrier ~ MeV -- yet stars burn\n")
    print(f"  {'reaction':>12}{'Z1 Z2':>7}{'T (K)':>10}{'E_G (keV)':>12}"
          f"{'peak E0':>11}")
    print("  " + "-" * 52)
    cases = [
        ("p + p", 1, 1, reduced_mass(M_P, M_P), 1.5e7),
        ("p + N14", 1, 7, reduced_mass(M_P, 14 * M_P), 1.5e7),
        ("He + He", 2, 2, reduced_mass(4 * M_P, 4 * M_P), 1e8),
        ("C + C", 6, 6, reduced_mass(12 * M_P, 12 * M_P), 5e8),
    ]
    for name, z1, z2, mu, T in cases:
        eg = gamow_energy(z1, z2, mu) / KEV
        e0 = peak_energy_kev(z1, z2, mu, T)
        print(f"  {name:>12}{z1*z2:>7}{T:>10.1e}{eg:>12.0f}{e0:>10.1f}k")

    print("\n  Fusion lives in a narrow window far out on the thermal tail: the Sun's")
    print("  p-p peak sits at ~6 keV, several times the mean 1.3 keV, where enough")
    print("  fast protons meet a high-enough tunnelling chance. Because the peak")
    print("  climbs steeply with nuclear charge, carbon burning needs ~500 million K")
    print("  while hydrogen ignites at 15 million -- the thermostat of stellar life.")

    _svg(os.path.join(outdir, "gamow.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'gamow.svg')}")


def _svg(path, size=720, pad=70, T=1.5e7):
    E_G = gamow_energy(1, 1, MU_PP)
    kT = K_B * T
    E0 = gamow_peak_energy(1, 1, MU_PP, T)
    Emax = 25.0 * KEV
    Es = [Emax * i / 400 for i in range(1, 401)]
    boltz = [math.exp(-E / kT) for E in Es]
    tunnel = [math.exp(-math.sqrt(E_G / E)) for E in Es]
    prod = [b * t for b, t in zip(boltz, tunnel)]
    # normalize each series to its own max for display on one axis
    def norm(xs):
        mx = max(xs)
        return [x / mx for x in xs]
    boltz_n, tunnel_n, prod_n = norm(boltz), norm(tunnel), norm(prod)

    def sx(E):
        return pad + E / Emax * (size - 2 * pad)

    def sy(y):
        return size - pad - y * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    def poly(ys):
        return " ".join(f"{sx(Es[i]):.1f},{sy(ys[i]):.1f}" for i in range(len(Es)))
    parts.append(f'<polyline points="{poly(boltz_n)}" fill="none" stroke="#ff6b6b" stroke-width="1.8"/>')
    parts.append(f'<polyline points="{poly(tunnel_n)}" fill="none" stroke="#4dabf7" stroke-width="1.8"/>')
    parts.append(f'<polyline points="{poly(prod_n)}" fill="none" stroke="#ffd43b" stroke-width="2.8"/>')

    # mark E0
    x0 = sx(E0)
    parts.append(f'<line x1="{x0:.1f}" y1="{pad}" x2="{x0:.1f}" y2="{size-pad}" '
                 f'stroke="#8b949e" stroke-width="1" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{x0+4:.1f}" y="{pad+40:.1f}" fill="#8b949e" '
                 f'font-size="10">Gamow peak E0 = {E0/KEV:.1f} keV</text>')

    parts.append(f'<text x="{pad+10}" y="{pad+22}" fill="#ff6b6b" font-size="12">'
                 f'Boltzmann tail exp(-E/kT)</text>')
    parts.append(f'<text x="{pad+10}" y="{pad+38}" fill="#4dabf7" font-size="12">'
                 f'tunnelling exp(-sqrt(E_G/E))</text>')
    parts.append(f'<text x="{pad+10}" y="{pad+54}" fill="#ffd43b" font-size="12">'
                 f'product = Gamow peak</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'The Gamow peak (proton-proton, solar core)</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">energy (keV, each curve self-normalized) -&gt;</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
