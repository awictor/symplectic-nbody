"""Demo: the Duffing oscillator -- a nonlinear spring and its double well.

Prints the regime and well structure across parameters and the backbone frequency bend, then
draws the double-well potential with a damped trajectory rolling into one well, and the
amplitude-frequency backbone that leans over (hardening vs softening).

    python examples/duffing_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from duffing import (potential, well_minima, regime, trajectory,  # noqa: E402
                     backbone_frequency)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Duffing: x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t)\n")
    print(f"  {'alpha':>7}{'beta':>7}{'regime':>14}{'minima':>18}")
    for alpha, beta in ((1.0, 1.0), (1.0, -0.5), (-1.0, 1.0), (1.0, 0.0)):
        mins = well_minima(alpha, beta)
        ms = ", ".join(f"{m:+.2f}" for m in mins)
        print(f"  {alpha:>7.1f}{beta:>7.1f}{regime(alpha, beta):>14}{ms:>18}")

    print("\n  Backbone (amplitude-dependent resonance frequency, hardening spring):")
    for A in (0.0, 0.5, 1.0, 1.5):
        print(f"    amplitude {A:.1f}  ->  omega_res = {backbone_frequency(A, 1.0, 1.0):.3f}")

    print("\n  The cubic term makes the spring's resonance bend: its peak frequency shifts with")
    print("  amplitude, so the response is multi-valued and JUMPS between branches as you sweep")
    print("  the drive (hysteresis). With alpha<0 the potential is a double well -- a buckled")
    print("  beam -- and a damped mass rolls into one side; driven hard, the Duffing goes chaotic.")

    _svg(os.path.join(outdir, "duffing.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'duffing.svg')}")


def _svg(path, size=720, pad=72):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">The Duffing oscillator</text>',
        f'<text x="20" y="48" fill="#8b949e" font-size="12">'
        f'the double-well potential with a mass rolling into a well (left); the resonance backbone (right)</text>',
    ]

    x0, x1 = pad, size - pad
    mid = size * 0.52

    # --- left: double-well potential + trajectory ---
    lx0, lx1 = pad, size * 0.50
    ly0, ly1 = size - pad, pad + 40
    alpha, beta = -1.0, 1.0
    xr = 1.7
    def PX(x):
        return (lx0 + lx1) / 2 + x / xr * (lx1 - lx0) / 2
    Vmax = potential(xr, alpha, beta)
    Vmin = potential(1.0, alpha, beta)
    def PY(V):
        return ly0 - (V - Vmin) / (Vmax - Vmin) * (ly0 - ly1)
    pts = []
    for i in range(121):
        x = -xr + 2 * xr * i / 120
        pts.append(f"{PX(x):.1f},{PY(potential(x, alpha, beta)):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#8b949e" stroke-width="2"/>')
    # wells
    for m in well_minima(alpha, beta):
        parts.append(f'<circle cx="{PX(m):.1f}" cy="{PY(potential(m, alpha, beta)):.1f}" r="3" fill="#4dabf7"/>')
    # damped trajectory rolling in: overlay x(t) as a settling marker path
    tr = trajectory((0.3, 0.0), 0.02, 3000, delta=0.25, alpha=alpha, beta=beta)
    tp = []
    for p in tr[::10]:
        tp.append(f"{PX(p[0]):.1f},{PY(potential(p[0], alpha, beta)):.1f}")
    parts.append(f'<polyline points="{" ".join(tp)}" fill="none" stroke="#06d6a0" '
                 f'stroke-width="1" opacity="0.6"/>')
    endx = tr[-1][0]
    parts.append(f'<circle cx="{PX(endx):.1f}" cy="{PY(potential(endx, alpha, beta)):.1f}" r="5" fill="#06d6a0"/>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">V(x) double well; mass settles in one well</text>')

    # --- right: backbone curves (amplitude vs resonance frequency) ---
    rx0, rx1 = size * 0.56, size - pad
    ry0, ry1 = size - pad, pad + 40
    A_max = 2.0
    f_lo, f_hi = 0.4, 2.2
    def AY(A):
        return ry0 - A / A_max * (ry0 - ry1)
    def FX(f):
        return rx0 + (f - f_lo) / (f_hi - f_lo) * (rx1 - rx0)
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.4"/>')
    for label, beta_b, col in (("hardening (beta>0)", 1.0, "#ff922b"),
                               ("linear (beta=0)", 0.0, "#8b949e"),
                               ("softening (beta<0)", -0.4, "#06d6a0")):
        bp = []
        A = 0.0
        while A <= A_max:
            f = backbone_frequency(A, 1.0, beta_b)
            if f_lo <= f <= f_hi:
                bp.append(f"{FX(f):.1f},{AY(A):.1f}")
            A += 0.03
        parts.append(f'<polyline points="{" ".join(bp)}" fill="none" stroke="{col}" stroke-width="2"/>')
    parts.append(f'<text x="{rx0+10:.1f}" y="{ry1+2:.1f}" fill="#ff922b" font-size="10">hardening -&gt;</text>')
    parts.append(f'<text x="{rx0+10:.1f}" y="{ry1+16:.1f}" fill="#06d6a0" font-size="10">&lt;- softening</text>')
    for f in (0.5, 1.0, 1.5, 2.0):
        parts.append(f'<text x="{FX(f):.1f}" y="{ry0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{f:.1f}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+28:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">resonance frequency (backbone leans with amplitude)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
