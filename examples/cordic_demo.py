"""Demo: CORDIC computing transcendental functions with only shifts and adds.

Shows CORDIC reproducing cos/sin/atan2/exp/ln/sqrt to near machine precision, and visualizes the
iterative rotation converging on a target angle. Draws the rotation path on the unit circle.

    python examples/cordic_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cordic import (cordic_cos, cordic_sin, cordic_atan2, cordic_hypot,
                    cordic_exp, cordic_ln, cordic_sqrt, _ATAN_TABLE, _INV_K)  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("CORDIC: sin, cos, exp, ln, sqrt using only shifts and additions\n")

    print("  Circular functions (vs math library):")
    for a in [0.5, 1.0, math.pi / 3]:
        print(f"    cos({a:.4f}) = {cordic_cos(a):.10f}  (math {math.cos(a):.10f})")
        print(f"    sin({a:.4f}) = {cordic_sin(a):.10f}  (math {math.sin(a):.10f})")

    print("\n  Vectoring mode (rectangular -> polar):")
    print(f"    atan2(1, 1) = {cordic_atan2(1, 1):.10f}  (pi/4 = {math.pi/4:.10f})")
    print(f"    hypot(3, 4) = {cordic_hypot(3, 4):.10f}  (= 5)")

    print("\n  Hyperbolic mode (exp, ln, sqrt):")
    print(f"    exp(2)   = {cordic_exp(2):.10f}  (math {math.exp(2):.10f})")
    print(f"    ln(10)   = {cordic_ln(10):.10f}  (math {math.log(10):.10f})")
    print(f"    sqrt(50) = {cordic_sqrt(50):.10f}  (math {math.sqrt(50):.10f})")

    # show the rotation converging: track the running angle after each step for a target
    target = 1.1
    x, y, z = _INV_K, 0.0, target
    path = [(x, y)]
    angles = []
    for i in range(12):
        d = 1.0 if z >= 0 else -1.0
        nx = x - d * (y * 2.0 ** -i)
        ny = y + d * (x * 2.0 ** -i)
        x, y = nx, ny
        z -= d * _ATAN_TABLE[i]
        path.append((x, y))
        angles.append(math.atan2(y, x))

    print(f"\n  Rotation converging on angle {target} rad (each step is a shift-add):")
    for i in range(0, 12, 2):
        print(f"    after {i+1:2d} steps: angle {angles[i]:.6f}  (error {abs(angles[i]-target):.2e})")

    print("\n  Each iteration rotates by +/- arctan(2^-i); because tan of the step is a power of two,")
    print("  the rotation is a bit shift, not a multiply. The running angle homes in on the target,")
    print("  and the final x,y are cos and sin (after dividing out the fixed CORDIC gain).")

    _svg(os.path.join(outdir, "cordic.svg"), path, target)
    print(f"\n  wrote {os.path.join(outdir, 'cordic.svg')}")


def _svg(path, rot_path, target, width=760, height=430):
    cx, cy, r = 250, 230, 160

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'CORDIC: iterative rotation homing in on cos/sin</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'each step rotates by +/- arctan(2^-i) (a bit shift); the path spirals to the target angle</text>',
    ]

    # unit circle
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#30363d" stroke-width="1.5"/>')
    parts.append(f'<line x1="{cx-r-20}" y1="{cy}" x2="{cx+r+20}" y2="{cy}" stroke="#30363d"/>')
    parts.append(f'<line x1="{cx}" y1="{cy-r-20}" x2="{cx}" y2="{cy+r+20}" stroke="#30363d"/>')

    # target ray
    tx = cx + (r + 15) * math.cos(target)
    ty = cy - (r + 15) * math.sin(target)
    parts.append(f'<line x1="{cx}" y1="{cy}" x2="{tx:.1f}" y2="{ty:.1f}" '
                 f'stroke="#ffd43b" stroke-width="1.5" stroke-dasharray="5,3"/>')
    parts.append(f'<text x="{tx+4:.0f}" y="{ty:.0f}" fill="#ffd43b" font-size="11">target {target} rad</text>')

    # rotation path (normalize each point to unit length for display)
    pts = []
    for (x, y) in rot_path:
        L = math.hypot(x, y) or 1
        px = cx + r * x / L
        py = cy - r * y / L
        pts.append(f"{px:.1f},{py:.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="1.6"/>')
    for i, (x, y) in enumerate(rot_path):
        L = math.hypot(x, y) or 1
        px = cx + r * x / L
        py = cy - r * y / L
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="#4dabf7"/>')

    # final point highlighted
    fx, fy = rot_path[-1]
    L = math.hypot(fx, fy) or 1
    parts.append(f'<circle cx="{cx + r*fx/L:.1f}" cy="{cy - r*fy/L:.1f}" r="6" fill="#06d6a0"/>')

    # value readout
    tx0 = 500
    parts.append(f'<text x="{tx0}" y="120" fill="#e6edf3" font-size="13">Only shift + add:</text>')
    parts.append(f'<text x="{tx0}" y="150" fill="#4dabf7" font-size="12">cos = {math.cos(target):.6f}</text>')
    parts.append(f'<text x="{tx0}" y="172" fill="#4dabf7" font-size="12">sin = {math.sin(target):.6f}</text>')
    parts.append(f'<text x="{tx0}" y="205" fill="#8b949e" font-size="11">no multiply, no divide,</text>')
    parts.append(f'<text x="{tx0}" y="222" fill="#8b949e" font-size="11">no function lookup</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
