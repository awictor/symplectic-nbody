"""Demo: Rabi oscillations -- a two-level atom flopping in a drive field.

Prints the pi/pi-2 pulse times and the resonance peak height versus detuning, then draws the
excited-state probability oscillating in time for several detunings (full contrast on
resonance, faster and shallower off it) plus the Lorentzian resonance curve.

    python examples/rabi_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rabi import (excited_probability, generalized_rabi, peak_probability,  # noqa: E402
                  pi_pulse_time, half_pi_pulse_time)


OMEGA = 2 * math.pi * 1e6   # 1 MHz Rabi frequency (rad/s)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Rabi oscillations: P_e(t) = (O^2/O_R^2) sin^2(O_R t/2), O = 2pi x 1 MHz\n")
    print("  Pulses on resonance:")
    print(f"    pi pulse  (full inversion, X gate): {pi_pulse_time(OMEGA)*1e9:.0f} ns")
    print(f"    pi/2 pulse (equal superposition):   {half_pi_pulse_time(OMEGA)*1e9:.0f} ns")

    print("\n  Detuning kills contrast (peak excitation P_max = O^2/(O^2+d^2)):")
    print(f"  {'detuning':>16}{'gen. Rabi':>14}{'peak P_e':>12}")
    for d_mhz in (0, 0.5, 1.0, 2.0):
        d = 2 * math.pi * d_mhz * 1e6
        print(f"  {d_mhz:>10.1f} MHz{generalized_rabi(OMEGA, d)/(2*math.pi*1e6):>10.2f} MHz"
              f"{peak_probability(OMEGA, d):>12.3f}")

    print("\n  On resonance the atom swings all the way to the excited state and back; detuned,")
    print("  it oscillates faster but only part-way. A pi pulse flips a qubit, a pi/2 pulse")
    print("  builds a superposition -- the elementary gates of atomic clocks and quantum bits.")

    _svg(os.path.join(outdir, "rabi.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'rabi.svg')}")


def _svg(path, size=720, pad=72):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Rabi oscillations</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'excited-state probability flops in time (top); Lorentzian resonance in detuning (bottom)</text>',
    ]

    x0, x1 = pad, size - pad
    mid = size * 0.54

    # --- top: P_e(t) for several detunings ---
    ty0, ty1 = mid - 26, pad + 44
    t_max = 3.0 * pi_pulse_time(OMEGA) * 2   # a few full cycles
    def TX(t):
        return x0 + t / t_max * (x1 - x0)
    def PY(P):
        return ty0 - P * (ty0 - ty1)
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x1}" y2="{ty0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x0}" y2="{ty1}" stroke="#8b949e" stroke-width="1.4"/>')
    for P in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{x0-6:.1f}" y="{PY(P)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{P:.1f}</text>')
        parts.append(f'<line x1="{x0}" y1="{PY(P):.1f}" x2="{x1}" y2="{PY(P):.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
    detunings = [(0.0, "#4dabf7", "on resonance"),
                 (2 * math.pi * 1e6, "#06d6a0", "detune 1 MHz"),
                 (2 * math.pi * 2e6, "#ff922b", "detune 2 MHz")]
    n = 300
    for d, col, lbl in detunings:
        pts = []
        for i in range(n + 1):
            t = t_max * i / n
            pts.append(f"{TX(t):.1f},{PY(excited_probability(OMEGA, d, t)):.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2.2"/>')
    # legend
    for i, (d, col, lbl) in enumerate(detunings):
        parts.append(f'<text x="{x0+90+i*180:.1f}" y="{ty1-6:.1f}" fill="{col}" font-size="10">{lbl}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{ty0+22:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">time -> (pi pulse fully inverts on resonance)</text>')

    # --- bottom: Lorentzian resonance ---
    by0, by1 = size - pad, mid + 40
    d_max = 2 * math.pi * 4e6
    def DX(d):
        return (x0 + x1) / 2 + d / d_max * (x1 - x0) / 2
    def RY(P):
        return by0 - P * (by0 - by1)
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x1}" y2="{by0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{(x0+x1)/2:.1f}" y1="{by0}" x2="{(x0+x1)/2:.1f}" y2="{by1}" '
                 f'stroke="#21262d" stroke-width="1"/>')
    pts = []
    for i in range(201):
        d = -d_max + 2 * d_max * i / 200
        pts.append(f"{DX(d):.1f},{RY(peak_probability(OMEGA, d)):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#8338ec" stroke-width="2.6"/>')
    # half-max markers at detuning = +/- Omega
    for d in (-OMEGA, OMEGA):
        parts.append(f'<circle cx="{DX(d):.1f}" cy="{RY(0.5):.1f}" r="4" fill="#ffd43b"/>')
    parts.append(f'<text x="{DX(OMEGA)+6:.1f}" y="{RY(0.5)-6:.1f}" fill="#ffd43b" font-size="10">'
                 f'half-max at detuning = Omega</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{by0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">detuning delta (peak excitation, Lorentzian of width Omega)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
