"""Demo: Fick diffusion -- a point release spreading as sqrt(t).

Prints diffusion lengths and times across scales (cell to room) and the Stokes-Einstein
coefficient, then draws a point release spreading into ever-wider, ever-lower Gaussians at
a sequence of times -- the diffusive sqrt(t) fan, with the rms-width markers.

    python examples/diffusion_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffusion import (gaussian_profile, rms_spread, diffusion_length,  # noqa: E402
                       diffusion_time, stokes_einstein)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    D = 1e-9   # small molecule in water (m^2/s)
    print("Fick diffusion: a random walk spreads as sqrt(t), not t\n")
    print("  Small molecule in water, D = 1e-9 m^2/s\n")
    print(f"  {'distance':>14}{'diffusion time':>20}")
    for L, lbl in ((1e-6, "1 um (organelle)"), (10e-6, "10 um (cell)"),
                   (1e-3, "1 mm (tissue)"), (1e-2, "1 cm"), (1.0, "1 m (room)")):
        t = diffusion_time(L, D)
        print(f"  {lbl:>14}{_fmt_time(t):>20}")

    print("\n  Spread of a point release (sigma = sqrt(2 D t)):")
    for t in (1.0, 100.0, 1e4):
        print(f"    t = {t:>8.0f} s  ->  sigma = {rms_spread(t, D)*1e3:.3f} mm")

    Dse = stokes_einstein(298.0, 1e-3, 1e-9)
    print("\n  Stokes-Einstein for a 1 nm sphere in water at 25 C: D = %.2e m^2/s" % Dse)
    print("  Diffusion is quick across a cell but takes ~30 years across a room, since time")
    print("  scales as L^2/D -- why microscopic life relies on it and large bodies need flow.")

    _svg(os.path.join(outdir, "diffusion.svg"), D)
    print(f"\n  wrote {os.path.join(outdir, 'diffusion.svg')}")


def _fmt_time(t):
    if t < 1e-3:
        return f"{t*1e6:.1f} us"
    if t < 1.0:
        return f"{t*1e3:.1f} ms"
    if t < 3600:
        return f"{t:.2f} s"
    if t < 86400 * 2:
        return f"{t/3600:.1f} hr"
    if t < 86400 * 800:
        return f"{t/86400:.0f} days"
    return f"{t/(86400*365.25):.0f} yr"


def _svg(path, D, size=720, pad=76):
    times = [(3.0, "#8338ec", "t"), (12.0, "#4dabf7", "4t"),
             (48.0, "#06d6a0", "16t"), (192.0, "#ff922b", "64t")]
    xmax = 4.0 * rms_spread(times[-1][0], D)      # domain from widest cloud
    cmax = gaussian_profile(0.0, times[0][0], D)  # tallest peak (earliest time)

    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 40
    cx = (x0 + x1) / 2.0

    def X(x):
        return cx + x / xmax * (x1 - x0) / 2.0

    def Y(c):
        return y0 - c / cmax * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Diffusion of a point release</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'each cloud stays a Gaussian; width grows as sqrt(t), height falls to conserve area</text>',
    ]

    # axes
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y1}" '
                 f'stroke="#21262d" stroke-width="1"/>')

    n = 200
    for t, col, lbl in times:
        pts = []
        for k in range(n + 1):
            x = -xmax + 2.0 * xmax * k / n
            pts.append(f"{X(x):.1f},{Y(gaussian_profile(x, t, D)):.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        # rms-width tick at +sigma
        sig = rms_spread(t, D)
        cs = gaussian_profile(sig, t, D)
        parts.append(f'<circle cx="{X(sig):.1f}" cy="{Y(cs):.1f}" r="3.5" fill="{col}"/>')
        peak = gaussian_profile(0.0, t, D)
        parts.append(f'<text x="{X(0)+6:.1f}" y="{Y(peak)-4:.1f}" fill="{col}" font-size="11">{lbl}</text>')

    # x scale label
    parts.append(f'<text x="{x1:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end">position (dots mark +1 sigma = sqrt(2Dt))</text>')
    parts.append(f'<text x="{cx:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">0</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
