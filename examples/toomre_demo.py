"""Demo: the Toomre Q stability of a galactic disk.

Prints Q for the solar neighbourhood (gas and stars) and the Toomre wavelength,
then draws Q vs galactocentric radius for a model Milky-Way disk, shading the
Q < 1 unstable band so the marginally-stable star-forming zone stands out.

    python examples/toomre_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from toomre import (toomre_q_gas, toomre_q_stars, is_stable,  # noqa: E402
                    toomre_wavelength, epicyclic_frequency_flat, PC, M_SUN)

V_CIRC = 220e3        # flat rotation curve, 220 km/s
C_S = 8e3             # gas sound speed, 8 km/s
SIGMA_R = 30e3        # stellar radial dispersion, 30 km/s


def sigma_gas(R_kpc):
    """Exponential gas surface density, ~13 Msun/pc^2 at the Sun (R=8 kpc)."""
    return 13.0 * M_SUN / PC ** 2 * math.exp(-(R_kpc - 8.0) / 6.0)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Toomre Q: rotation + pressure vs self-gravity in a disk\n")
    print("  Q = c_s kappa / (pi G Sigma)   [gas]")
    print("  Q = sigma_R kappa / (3.36 G Sigma)  [stars]\n")

    R = 8000.0 * PC
    kappa = epicyclic_frequency_flat(V_CIRC, R)
    Sg = sigma_gas(8.0)
    Ss = 50.0 * M_SUN / PC ** 2
    Qg = toomre_q_gas(C_S, kappa, Sg)
    Qs = toomre_q_stars(SIGMA_R, kappa, Ss)
    lam = toomre_wavelength(Sg, kappa) / PC / 1000.0

    print(f"  solar neighbourhood (R = 8 kpc):")
    print(f"    gas Q     = {Qg:.2f}  ({'stable' if is_stable(Qg) else 'UNSTABLE'})")
    print(f"    stellar Q = {Qs:.2f}  ({'stable' if is_stable(Qs) else 'UNSTABLE'})")
    print(f"    Toomre wavelength = {lam:.2f} kpc (the scale of the structures)\n")

    print(f"  {'R (kpc)':>9}{'kappa (/Gyr)':>14}{'gas Q':>9}{'state':>12}")
    print("  " + "-" * 44)
    for R_kpc in (2, 4, 6, 8, 10, 14, 18):
        Rm = R_kpc * 1000.0 * PC
        k = epicyclic_frequency_flat(V_CIRC, Rm)
        Q = toomre_q_gas(C_S, k, sigma_gas(R_kpc))
        state = "stable" if is_stable(Q) else "UNSTABLE"
        print(f"  {R_kpc:>9.0f}{k*3.156e16:>14.1f}{Q:>9.2f}{state:>12}")

    print("\n  The Milky Way sits at Q ~ 1.5-2 -- marginally stable. That is no")
    print("  accident: a disk that cools below Q = 1 fragments into clumps and")
    print("  spiral arms, and the resulting star formation heats it back up, so")
    print("  disks self-regulate to hover near the stability line.")

    _svg(os.path.join(outdir, "toomre.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'toomre.svg')}")


def _svg(path, size=720, pad=66):
    R_kpc = [1.0 + 0.3 * i for i in range(0, 60)]  # 1..~19 kpc
    Qs = []
    for R in R_kpc:
        k = epicyclic_frequency_flat(V_CIRC, R * 1000.0 * PC)
        Qs.append(toomre_q_gas(C_S, k, sigma_gas(R)))
    xmin, xmax = R_kpc[0], R_kpc[-1]
    ymax = max(3.0, max(Qs) * 1.05)
    ymin = 0.0

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # unstable band Q < 1 shaded red
    y1 = sy(1.0)
    parts.append(f'<rect x="{pad}" y="{y1:.1f}" width="{size-2*pad}" '
                 f'height="{size-pad-y1:.1f}" fill="#ff6b6b" fill-opacity="0.12"/>')
    parts.append(f'<line x1="{pad}" y1="{y1:.1f}" x2="{size-pad}" y2="{y1:.1f}" '
                 f'stroke="#ff6b6b" stroke-width="1.4" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{pad+8}" y="{y1+18:.1f}" fill="#ff6b6b" '
                 f'font-size="11">Q &lt; 1: unstable, fragments into clumps/arms</text>')

    parts.append(f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>')

    poly = " ".join(f"{sx(R_kpc[i]):.1f},{sy(Qs[i]):.1f}" for i in range(len(R_kpc)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4dabf7" stroke-width="2.4"/>')

    # mark the Sun at 8 kpc
    ks = epicyclic_frequency_flat(V_CIRC, 8000 * PC)
    Qsun = toomre_q_gas(C_S, ks, sigma_gas(8.0))
    parts.append(f'<circle cx="{sx(8.0):.1f}" cy="{sy(Qsun):.1f}" r="5" fill="#ffd43b"/>')
    parts.append(f'<text x="{sx(8.0)+8:.1f}" y="{sy(Qsun)+4:.1f}" fill="#ffd43b" '
                 f'font-size="11">Sun (Q = {Qsun:.1f})</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Toomre Q across a galactic disk</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'the Milky Way hovers just above Q = 1: self-regulated marginal stability</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">galactocentric radius (kpc) -&gt;</text>')
    parts.append(f'<text x="{pad-12}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'gas Toomre Q</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
