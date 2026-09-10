"""Demo: the Aharonov-Bohm effect -- interference shifted by an untouched flux.

Prints the flux quanta and phase shifts for a range of enclosed fluxes, then draws two
figures: the interference fringes sliding as flux threads the (field-free-on-path) solenoid,
and the phase winding linearly with flux, periodic in the flux quantum.

    python examples/aharonov_bohm_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aharonov_bohm import (phase_shift, flux_quantum, num_flux_quanta,  # noqa: E402
                           fringe_shift, field_for_one_quantum, FLUX_QUANTUM_SC)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    phi0 = flux_quantum()
    print("Aharonov-Bohm: delta_phi = q Phi / hbar, from flux Phi with B=0 on the path\n")
    print("  Electron flux quantum Phi_0 = h/e = %.3e Wb\n" % phi0)
    print(f"  {'flux / Phi_0':>14}{'phase (rad)':>14}{'fringe shift':>14}")
    for m in (0.0, 0.25, 0.5, 1.0, 2.5):
        flux = m * phi0
        print(f"  {m:>14.2f}{phase_shift(flux):>14.3f}{fringe_shift(flux):>14.2f}")

    print("\n  Superconducting flux quantum h/2e = %.3e Wb (Cooper pairs)." % FLUX_QUANTUM_SC)
    print("  One quantum through a 1 mm^2 SQUID loop needs only %.2e T -- how SQUIDs sense"
          % field_for_one_quantum(1e-6, 2 * 1.602176634e-19))
    print("  fields a billion times weaker than Earth's. The electron never sees the field,")
    print("  only the vector potential -- proof the potentials are physically real in QM.")

    _svg(os.path.join(outdir, "aharonov_bohm.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'aharonov_bohm.svg')}")


def _svg(path, size=720, pad=76):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The Aharonov-Bohm effect</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'enclosed flux slides the interference fringes (top) though B = 0 on the electron path</text>',
    ]

    x0, x1 = pad, size - pad
    mid = size * 0.54

    # --- top: fringe intensity for a few flux values ---
    ty0, ty1 = mid - 26, pad + 44
    def IX(x):
        return x0 + (x + 6) / 12 * (x1 - x0)   # x in [-6, 6] (screen position, arb units)
    fluxes = [(0.0, "#4dabf7", "Phi = 0"),
              (0.5, "#06d6a0", "Phi = 0.5 Phi_0"),
              (1.0, "#ff922b", "Phi = Phi_0")]
    band_h = (ty0 - ty1) / 3
    for row, (m, col, lbl) in enumerate(fluxes):
        base = ty1 + band_h * (row + 0.5)
        dphi = 2 * math.pi * m
        n = 240
        pts = []
        for i in range(n + 1):
            x = -6 + 12 * i / n
            I = 0.5 * (1 + math.cos(2 * math.pi * x + dphi))   # shifted fringes
            pts.append(f"{IX(x):.1f},{base - I * band_h * 0.8:.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2"/>')
        parts.append(f'<text x="{x0:.1f}" y="{base - band_h*0.85:.1f}" fill="{col}" font-size="10">{lbl}</text>')
    # dashed line marking a fixed screen position to show the shift
    parts.append(f'<line x1="{IX(0):.1f}" y1="{ty1:.1f}" x2="{IX(0):.1f}" y2="{ty0:.1f}" '
                 f'stroke="#8b949e" stroke-width="0.8" stroke-dasharray="4 4" opacity="0.5"/>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{ty0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">screen position -> (fringes slide by one period per flux quantum)</text>')

    # --- bottom: phase vs flux (linear, periodic markers) ---
    phi0 = flux_quantum()
    by0, by1 = size - pad, mid + 40
    m_max = 3.0
    def FX(m):
        return x0 + m / m_max * (x1 - x0)
    def PY(p):
        return by0 - p / (2 * math.pi * m_max) * (by0 - by1)
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x1}" y2="{by0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x0}" y2="{by1}" stroke="#8b949e" stroke-width="1.4"/>')
    pts = " ".join(f"{FX(m_max*i/100):.1f},{PY(phase_shift(m_max*i/100*phi0)):.1f}" for i in range(101))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#8338ec" stroke-width="2.6"/>')
    # 2 pi markers at each integer flux quantum
    for m in range(1, 4):
        parts.append(f'<line x1="{FX(m):.1f}" y1="{by0:.1f}" x2="{FX(m):.1f}" y2="{PY(2*math.pi*m):.1f}" '
                     f'stroke="#ffd43b" stroke-width="0.8" stroke-dasharray="3 3" opacity="0.6"/>')
        parts.append(f'<circle cx="{FX(m):.1f}" cy="{PY(2*math.pi*m):.1f}" r="3.5" fill="#ffd43b"/>')
        parts.append(f'<text x="{FX(m):.1f}" y="{by0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{m}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{by0+28:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">enclosed flux (units of Phi_0); phase = 2 pi per quantum</text>')
    parts.append(f'<text x="{x0+90:.1f}" y="{by1+2:.1f}" fill="#8338ec" font-size="10">'
                 f'AB phase delta_phi = q Phi / hbar</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
