"""Demo: gravitational lensing -- the Einstein ring and the microlensing curve.

Reproduces the 1.75-arcsec solar deflection, draws the two images and Einstein
ring for a point-mass lens, and renders a microlensing light curve (total
magnification as a source drifts past the lens) to SVG.

    python examples/lensing_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lensing import (deflection_angle, image_positions, total_magnification)  # noqa: E402

_BARS = " .:-=+*#@"
M_SUN = 1.989e30
R_SUN = 6.96e8


def spark(vals, lo, hi):
    span = (hi - lo) or 1.0
    return "".join(_BARS[max(0, min(8, int((v - lo) / span * 8)))] for v in vals)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    alpha = math.degrees(deflection_angle(M_SUN, R_SUN)) * 3600.0
    print("Gravitational lensing\n")
    print(f"  light deflection at the Sun's limb : {alpha:.3f} arcsec "
          f"(Eddington 1919 measured ~1.75)\n")

    # microlensing light curve: source drifts past lens with impact param u0
    tE = 1.0
    u0 = 0.2
    us = [-2.0 + 4.0 * k / 199 for k in range(200)]
    mags = [total_magnification(math.hypot(u, u0), tE) for u in us]
    print(f"  microlensing light curve (min impact u0={u0}):")
    print("  " + spark(mags, min(mags), max(mags)))
    print(f"  peak magnification A_max = {max(mags):.2f} at closest approach\n")

    print("  A point-source drifting behind a mass brightens symmetrically then")
    print("  fades -- the achromatic, time-symmetric microlensing signature used")
    print("  to find exoplanets and dark compact objects.")

    _svg(us, mags, u0, tE, os.path.join(outdir, "lensing.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'lensing.svg')}")


def _svg(us, mags, u0, tE, path, size=720, pad=56):
    # left panel: light curve; right panel: ring + two images at closest approach
    umin, umax = us[0], us[-1]
    amax = max(mags)

    def sx(u):
        return pad + (u - umin) / (umax - umin) * (size / 2 - 1.5 * pad)

    def sy(a):
        return size - pad - (a - 1.0) / (amax - 1.0) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # light curve
    poly = " ".join(f"{sx(us[i]):.1f},{sy(mags[i]):.1f}" for i in range(len(us)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ffbe0b" stroke-width="1.8"/>')
    parts.append(f'<text x="{pad}" y="30" fill="#e6edf3" font-size="16">'
                 f'Microlensing light curve</text>')
    parts.append(f'<text x="{pad}" y="{size-16}" fill="#8b949e" font-size="11">'
                 f'total magnification vs source position (u0={u0})</text>')

    # right panel: Einstein ring + images
    cx, cy, R = 3 * size // 4, size // 2, size // 5

    def rx(x): return cx + x / (2.5 * tE) * R
    def ry(y): return cy - y / (2.5 * tE) * R

    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" '
                 f'stroke="#4cc9f0" stroke-dasharray="4,4"/>')  # Einstein ring
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#ffd166"/>')  # lens
    tp, tm = image_positions(u0, tE)
    # source at (0, u0), images along the line through source and lens (take y-axis)
    parts.append(f'<circle cx="{rx(0)}" cy="{ry(u0)}" r="3" fill="#e6edf3"/>')  # source
    parts.append(f'<circle cx="{rx(0)}" cy="{ry(tp)}" r="4" fill="#ff006e"/>')  # image +
    parts.append(f'<circle cx="{rx(0)}" cy="{ry(tm)}" r="4" fill="#ff006e"/>')  # image -
    parts.append(f'<text x="{cx-R}" y="{cy-R-12}" fill="#4cc9f0" font-size="12">'
                 f'Einstein ring &amp; two images</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
