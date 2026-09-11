"""Demo: the Van der Pol oscillator -- a self-sustaining rhythm.

Prints the limit-cycle amplitude and period across the nonlinearity mu, then draws the phase
portrait (two starts spiralling onto the same limit cycle) and the waveform morphing from
near-sinusoidal at small mu to relaxation oscillation at large mu.

    python examples/van_der_pol_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from van_der_pol import (trajectory, rk4_step, limit_cycle_amplitude,  # noqa: E402
                         relaxation_period, measured_period)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Van der Pol: x'' - mu(1-x^2)x' + x = 0 -- damping that changes sign with amplitude\n")
    print(f"  {'mu':>6}{'amplitude':>12}{'period':>12}{'character':>20}")
    for mu in (0.1, 0.5, 1.0, 3.0, 5.0):
        amp = limit_cycle_amplitude(mu)
        T = measured_period(mu)
        ch = "near-sinusoidal" if mu < 0.5 else ("relaxation" if mu > 2 else "transitional")
        print(f"  {mu:>6.1f}{amp:>12.2f}{T:>12.2f}{ch:>20}")

    print("\n  The amplitude settles near 2 for every mu -- the limit cycle forgets how it")
    print("  started, whether launched tiny (grows in) or huge (decays in). That self-regulated")
    print("  rhythm is the model for heartbeats, firing neurons, and bowed strings; at large mu")
    print("  it becomes relaxation oscillation, slow charges broken by fast jumps (period ~1.6 mu).")

    _svg(os.path.join(outdir, "van_der_pol.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'van_der_pol.svg')}")


def _svg(path, size=720, pad=70):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">The Van der Pol oscillator</text>',
        f'<text x="20" y="48" fill="#8b949e" font-size="12">'
        f'two starts spiral onto the same limit cycle (left); waveform vs nonlinearity mu (right)</text>',
    ]

    # --- left: phase portrait (x vs v), mu=1 ---
    lx0, lx1 = pad, size * 0.50
    lcx = (lx0 + lx1) / 2
    lcy = size * 0.46
    sc = (lx1 - lx0) / 2 / 3.2
    # start big (outside) and small (inside), both spiral to the cycle
    for start, col in (((3.0, 0.0), "#ff6b6b"), ((0.05, 0.0), "#06d6a0")):
        tr = trajectory(start, 0.01, 3000, mu=1.0)
        pts = " ".join(f"{lcx + p[0]*sc:.1f},{lcy - p[1]*sc:.1f}" for p in tr)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="0.8" opacity="0.75"/>')
    # the limit cycle itself, bright
    lc = trajectory((2.0, 0.0), 0.01, 4000, mu=1.0)[2000:]
    lcpts = " ".join(f"{lcx + p[0]*sc:.1f},{lcy - p[1]*sc:.1f}" for p in lc)
    parts.append(f'<polyline points="{lcpts}" fill="none" stroke="#4dabf7" stroke-width="2"/>')
    # axes
    parts.append(f'<line x1="{lx0:.1f}" y1="{lcy:.1f}" x2="{lx1:.1f}" y2="{lcy:.1f}" stroke="#21262d" stroke-width="1"/>')
    parts.append(f'<line x1="{lcx:.1f}" y1="{lcy-2.8*sc:.1f}" x2="{lcx:.1f}" y2="{lcy+2.8*sc:.1f}" stroke="#21262d" stroke-width="1"/>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{size-pad+4:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">phase space x vs x&#39; (red from outside, green from inside)</text>')

    # --- right: waveforms x(t) for small and large mu ---
    rx0, rx1 = size * 0.56, size - pad
    ry_top, ry_bot = size * 0.30, size * 0.70
    def draw_wave(mu, cy, col, label):
        tr = trajectory((0.1, 0.0), 0.02, 3000, mu=mu)[1000:]  # settled portion
        n = len(tr)
        pts = []
        for i, p in enumerate(tr[:1200]):
            x = rx0 + (rx1 - rx0) * i / 1200
            pts.append(f"{x:.1f},{cy - p[0] * 22:.1f}")
        parts.append(f'<line x1="{rx0:.1f}" y1="{cy:.1f}" x2="{rx1:.1f}" y2="{cy:.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="1.8"/>')
        parts.append(f'<text x="{rx0:.1f}" y="{cy - 52:.1f}" fill="{col}" font-size="11">{label}</text>')
    draw_wave(0.3, ry_top, "#06d6a0", "mu = 0.3: near-sinusoidal")
    draw_wave(5.0, ry_bot, "#ff922b", "mu = 5: relaxation (slow charge, fast jump)")
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{size-pad+4:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">x(t): small mu smooth, large mu spiky</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
