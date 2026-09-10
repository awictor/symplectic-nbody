"""Demo: the Josephson junction -- a tunneling supercurrent and the volt standard.

Prints the Josephson voltage-to-frequency conversion and the Shapiro-step voltages, then
draws two figures: the DC supercurrent I = I_c sin(phi) versus phase, and the irradiated
I-V curve climbing in quantized Shapiro voltage steps.

    python examples/josephson_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from josephson import (supercurrent, josephson_frequency, voltage_from_frequency,  # noqa: E402
                       josephson_constant, shapiro_step_voltage, coupling_energy)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Josephson junction: I = I_c sin(phi), f = 2eV/h = 483.6 GHz/mV\n")
    print("  Josephson constant K_J = 2e/h = %.4e Hz/V\n" % josephson_constant())
    print(f"  {'voltage':>10}{'Josephson freq':>18}")
    for V_uv in (10, 100, 1000):
        print(f"  {V_uv:>7} uV{josephson_frequency(V_uv*1e-6)/1e9:>15.2f} GHz")

    print("\n  Shapiro steps under 70 GHz irradiation (the volt standard):")
    for n in (1, 2, 5, 10):
        print(f"    step {n:>2}:  {shapiro_step_voltage(n, 70e9)*1e6:>7.2f} uV")

    print("\n  A 1 uA junction has coupling energy E_J = %.2e J (%.1f GHz x h)."
          % (coupling_energy(1e-6), coupling_energy(1e-6) / 6.626e-34 / 1e9))
    print("  Cooper pairs tunnel the barrier with zero voltage (DC effect); a DC voltage makes")
    print("  the phase wind and the current oscillate (AC effect). The exact V<->f link, tied")
    print("  only to e and h, is how the volt is now defined and how quantum voltmeters work.")

    _svg(os.path.join(outdir, "josephson.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'josephson.svg')}")


def _svg(path, size=720, pad=76):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The Josephson junction</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'DC supercurrent I = I_c sin(phi) (top); irradiated I-V climbing in Shapiro steps (bottom)</text>',
    ]

    x0, x1 = pad, size - pad
    mid = size * 0.54

    # --- top: I vs phase ---
    ty0, ty1 = mid - 30, pad + 44
    cy = (ty0 + ty1) / 2
    def PX(phi):
        return x0 + (phi + 2 * math.pi) / (4 * math.pi) * (x1 - x0)
    def IY(I):
        return cy - I * (ty0 - ty1) / 2 * 0.9
    parts.append(f'<line x1="{x0}" y1="{cy:.1f}" x2="{x1}" y2="{cy:.1f}" stroke="#21262d" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x0}" y2="{ty1}" stroke="#8b949e" stroke-width="1.4"/>')
    n = 240
    pts = []
    for i in range(n + 1):
        phi = -2 * math.pi + 4 * math.pi * i / n
        pts.append(f"{PX(phi):.1f},{IY(supercurrent(1.0, phi)):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')
    parts.append(f'<text x="{x0-6:.1f}" y="{IY(1)+3:.1f}" fill="#8b949e" font-size="9" text-anchor="end">+I_c</text>')
    parts.append(f'<text x="{x0-6:.1f}" y="{IY(-1)+3:.1f}" fill="#8b949e" font-size="9" text-anchor="end">-I_c</text>')
    for phi, lbl in ((-2*math.pi, "-2pi"), (0, "0"), (2*math.pi, "2pi")):
        parts.append(f'<text x="{PX(phi):.1f}" y="{ty0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{lbl}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{ty0+28:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">phase difference phi (zero-voltage DC supercurrent)</text>')

    # --- bottom: I-V staircase (Shapiro steps under 70 GHz) ---
    by0, by1 = size - pad, mid + 40
    f = 70e9
    dV = voltage_from_frequency(f)          # step spacing in volts
    nsteps = 6
    Vmax = (nsteps + 0.5) * dV
    Imax = 1.0
    def VX(V):
        return x0 + V / Vmax * (x1 - x0)
    def CY(I):
        return by0 - I / Imax * (by0 - by1)
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x1}" y2="{by0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x0}" y2="{by1}" stroke="#8b949e" stroke-width="1.4"/>')
    # staircase: vertical current rise at each step voltage, flat plateau between
    step_pts = [f"{VX(0):.1f},{CY(0):.1f}"]
    for k in range(1, nsteps + 1):
        Vk = k * dV
        Ik = k / (nsteps + 1)
        # plateau up to Vk then jump
        step_pts.append(f"{VX(Vk):.1f},{CY(Ik - 1.0/(nsteps+1)):.1f}")
        step_pts.append(f"{VX(Vk):.1f},{CY(Ik):.1f}")
    parts.append(f'<polyline points="{" ".join(step_pts)}" fill="none" stroke="#06d6a0" stroke-width="2.4"/>')
    for k in range(1, nsteps + 1):
        Vk = k * dV
        parts.append(f'<line x1="{VX(Vk):.1f}" y1="{by0:.1f}" x2="{VX(Vk):.1f}" y2="{by1:.1f}" '
                     f'stroke="#ffd43b" stroke-width="0.6" stroke-dasharray="3 3" opacity="0.4"/>')
    parts.append(f'<text x="{VX(dV):.1f}" y="{by0+14:.1f}" fill="#ffd43b" font-size="9" '
                 f'text-anchor="middle">{dV*1e6:.0f} uV</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{by0+28:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">voltage (quantized Shapiro steps at n h f / 2e, 70 GHz drive)</text>')
    parts.append(f'<text x="{x0+90:.1f}" y="{by1+2:.1f}" fill="#06d6a0" font-size="10">'
                 f'each plateau is an exact, constants-only voltage</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
