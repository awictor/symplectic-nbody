"""Demo: the Kerr black hole -- spin, frame-dragging, and the ISCO.

Shows how the innermost stable circular orbit depends on spin (prograde vs
retrograde), how black-hole spin is inferred from the disk's inner edge, and
renders the ISCO-vs-spin curves plus a horizon/ergosphere diagram to SVG.

    python examples/kerr_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kerr import (horizons, ergosphere_radius, isco_radius,  # noqa: E402
                  horizon_angular_velocity)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Kerr black hole: spin sets the horizon and the ISCO (units of M)\n")
    print(f"  {'a/M':>6}{'horizon':>10}{'ISCO pro':>11}{'ISCO retro':>12}{'Omega_H':>10}")
    print("  " + "-" * 49)
    spins = [0.0, 0.3, 0.6, 0.9, 0.99, 1.0]
    for a in spins:
        rp, _ = horizons(a)
        print(f"  {a:>6.2f}{rp:>10.3f}{isco_radius(a, True):>11.3f}"
              f"{isco_radius(a, False):>12.3f}{horizon_angular_velocity(a):>10.3f}")
    print("\n  A fast-spinning hole drags its prograde ISCO down toward the horizon")
    print("  (1M at extremal spin), so its accretion disk reaches deeper and radiates")
    print("  more efficiently. Measuring the disk's inner edge is how spins are found.")

    _svg(os.path.join(outdir, "kerr.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'kerr.svg')}")


def _svg(path, size=760, pad=60):
    # left: ISCO vs spin curves; right: horizon + ergosphere for a=0.9
    aa = [0.001 * i for i in range(0, 1000)]
    pro = [isco_radius(a, True) for a in aa]
    retro = [isco_radius(a, False) for a in aa]
    half = size // 2

    def sx(a):
        return pad + a * (half - 1.5 * pad)

    def sy(r):
        return size - pad - r / 10.0 * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{half-pad/2}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    ppoly = " ".join(f"{sx(aa[i]):.1f},{sy(pro[i]):.1f}" for i in range(len(aa)))
    rpoly = " ".join(f"{sx(aa[i]):.1f},{sy(retro[i]):.1f}" for i in range(len(aa)))
    parts.append(f'<polyline points="{ppoly}" fill="none" stroke="#4cc9f0" stroke-width="2"/>')
    parts.append(f'<polyline points="{rpoly}" fill="none" stroke="#ff006e" stroke-width="2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="15">ISCO vs spin</text>')
    parts.append(f'<text x="{pad+8}" y="{sy(1.5):.1f}" fill="#4cc9f0" font-size="11">prograde -&gt; 1M</text>')
    parts.append(f'<text x="{pad+8}" y="{sy(9.3):.1f}" fill="#ff006e" font-size="11">retrograde -&gt; 9M</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+22}" fill="#8b949e" font-size="11">a/M -&gt;</text>')

    # right panel: horizon + ergosphere for a=0.9 (equatorial slice)
    a = 0.9
    rp, _ = horizons(a)
    cx, cy, scale = 3 * size // 4, size // 2, (size // 2 - pad) / 3.0

    def rx(x): return cx + x * scale
    def ry(y): return cy - y * scale

    # ergosphere (theta-dependent): sample the static-limit surface
    ergo_pts = []
    for k in range(0, 361, 4):
        th = math.radians(k)
        r = ergosphere_radius(a, th)
        ergo_pts.append((r * math.sin(th), r * math.cos(th)))
    epoly = " ".join(f"{rx(x):.1f},{ry(y):.1f}" for x, y in ergo_pts)
    parts.append(f'<polygon points="{epoly}" fill="#f4a26133" stroke="#f4a261" stroke-width="1"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{rp*scale:.1f}" fill="#000000" stroke="#4cc9f0"/>')
    parts.append(f'<text x="{cx-2.5*scale:.0f}" y="{cy-2.6*scale:.0f}" fill="#e6edf3" '
                 f'font-size="14">a=0.9M: horizon + ergosphere</text>')
    parts.append(f'<text x="{cx-2.5*scale:.0f}" y="{cy+2.9*scale:.0f}" fill="#f4a261" '
                 f'font-size="11">orange = ergosphere (frame-dragging region)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
