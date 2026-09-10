"""Demo: tadpole and horseshoe coorbital orbits in the CR3BP rotating frame.

Integrates a Trojan-like tadpole orbit near L4 and a Janus/Epimetheus-like
horseshoe orbit that wraps around L3, and renders both paths in the rotating
frame to SVG (with the primary, secondary, and Lagrange points marked).

    python examples/coorbital_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cr3bp import CR3BP  # noqa: E402
from coorbital import (coorbital_trajectory, angular_range, classify,  # noqa: E402
                       start_near_L4, start_on_corotation)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    mu = 0.001
    m = CR3BP(mu)
    print("Coorbital motion in the CR3BP rotating frame (mu = 0.001)\n")

    x0, y0 = start_near_L4(m, offset=0.008)
    tx, ty, ta = coorbital_trajectory(m, x0, y0, dt=0.005, steps=150000, sample_every=60)
    print(f"  tadpole   (near L4): angular range {angular_range(ta):5.0f} deg -> {classify(ta)}")

    x0, y0 = start_on_corotation(180.0)
    hx, hy, ha = coorbital_trajectory(m, x0, y0, dt=0.005, steps=300000, sample_every=60)
    print(f"  horseshoe (near L3): angular range {angular_range(ha):5.0f} deg -> {classify(ha)}")

    print("\n  The tadpole loops one Lagrange point (like Jupiter's Trojans); the")
    print("  horseshoe wraps around L3 enclosing both L4 and L5, turning back before")
    print("  it reaches the planet (like Saturn's moons Janus & Epimetheus).")

    _svg(m, tx, ty, hx, hy, os.path.join(outdir, "coorbital.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'coorbital.svg')}")


def _svg(model, tx, ty, hx, hy, path, size=720, pad=40, extent=1.4):
    def sx(x): return size / 2 + x / extent * (size / 2 - pad)
    def sy(y): return size / 2 - y / extent * (size / 2 - pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # corotation circle
    parts.append(f'<circle cx="{sx(0):.1f}" cy="{sy(0):.1f}" '
                 f'r="{1.0/extent*(size/2-pad):.1f}" fill="none" '
                 f'stroke="#21262d" stroke-dasharray="3,4"/>')
    # horseshoe (grey), tadpole (teal)
    hpoly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(hx, hy))
    tpoly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(tx, ty))
    parts.append(f'<polyline points="{hpoly}" fill="none" stroke="#8b949e" '
                 f'stroke-width="0.8" stroke-opacity="0.8"/>')
    parts.append(f'<polyline points="{tpoly}" fill="none" stroke="#2a9d8f" '
                 f'stroke-width="1.2"/>')
    # primary, secondary, Lagrange points
    parts.append(f'<circle cx="{sx(-model.mu):.1f}" cy="{sy(0):.1f}" r="7" fill="#ffd166"/>')
    parts.append(f'<circle cx="{sx(1-model.mu):.1f}" cy="{sy(0):.1f}" r="4" fill="#e6edf3"/>')
    for name, (x, y) in model.lagrange_points().items():
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="#f4a261"/>')
        parts.append(f'<text x="{sx(x)+5:.1f}" y="{sy(y)-4:.1f}" fill="#f4a261" '
                     f'font-size="10">{name}</text>')
    parts.append(f'<text x="16" y="26" fill="#e6edf3" font-size="16">'
                 f'Coorbital orbits: tadpole (teal) &amp; horseshoe (grey)</text>')
    parts.append(f'<text x="16" y="{size-14}" fill="#8b949e" font-size="11">'
                 f'rotating frame; gold=primary, white=secondary, orange=Lagrange points</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
