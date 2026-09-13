"""Demo: PCHIP shape-preserving interpolation vs a natural cubic spline.

Interpolates step-like monotone data with both PCHIP and a natural cubic spline, showing the spline
overshoots (invents dips and bulges) while PCHIP stays monotone and inside the data range. Draws both
curves through the data.

    python examples/pchip_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pchip import PCHIP  # noqa: E402

# reuse the repo's natural cubic spline for the comparison
from spline import CubicSpline  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("PCHIP: shape-preserving cubic interpolation (no overshoot)\n")

    # classic overshoot-provoking data: a monotone step
    x = [0, 1, 2, 3, 4, 5, 6]
    y = [0, 0, 0, 1, 1, 1, 1]
    print(f"  Monotone step data: x={x}, y={y}\n")

    p = PCHIP(x, y)
    s = CubicSpline(x, y)

    xs = [x[0] + (x[-1] - x[0]) * k / 600 for k in range(601)]
    p_min = min(p(xq) for xq in xs)
    p_max = max(p(xq) for xq in xs)
    s_min = min(s(xq) for xq in xs)
    s_max = max(s(xq) for xq in xs)

    print(f"  {'method':>16}  {'min value':>10}  {'max value':>10}  {'overshoot?':>11}")
    print(f"  {'PCHIP':>16}  {p_min:>10.4f}  {p_max:>10.4f}  {'no' if p_min >= -1e-9 and p_max <= 1 + 1e-9 else 'YES':>11}")
    print(f"  {'cubic spline':>16}  {s_min:>10.4f}  {s_max:>10.4f}  {'no' if s_min >= -1e-9 and s_max <= 1 + 1e-9 else 'YES':>11}")

    # is PCHIP monotone?
    pv = [p(xq) for xq in xs]
    mono = all(pv[k + 1] >= pv[k] - 1e-9 for k in range(len(pv) - 1))
    sv = [s(xq) for xq in xs]
    smono = all(sv[k + 1] >= sv[k] - 1e-9 for k in range(len(sv) - 1))
    print(f"\n  PCHIP monotone: {mono}    cubic spline monotone: {smono}")
    print(f"  PCHIP dips below 0 by {max(0, -p_min):.4f}; spline dips below 0 by {max(0, -s_min):.4f}")
    print("\n  The cubic spline is smoother (C^2) but overshoots -- inventing wiggles the data never")
    print("  had. PCHIP gives up one derivative (C^1) to guarantee no overshoot on monotone data.")

    _svg(os.path.join(outdir, "pchip.svg"), x, y, p, s)
    print(f"\n  wrote {os.path.join(outdir, 'pchip.svg')}")


def _svg(path, x, y, p, s, width=760, height=410):
    xs = [x[0] + (x[-1] - x[0]) * k / 600 for k in range(601)]
    pv = [p(xq) for xq in xs]
    sv = [s(xq) for xq in xs]
    ymin = min(min(pv), min(sv), min(y)) - 0.1
    ymax = max(max(pv), max(sv), max(y)) + 0.1
    ox, oy, ow, oh = 50, 55, width - 90, height - 100

    def px(xq):
        return ox + ow * (xq - x[0]) / (x[-1] - x[0])

    def py(v):
        return oy + oh * (1 - (v - ymin) / (ymax - ymin))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Monotone step data: cubic spline (red) overshoots, PCHIP (green) does not</text>',
    ]
    # zero and one reference lines
    for lvl in (0.0, 1.0):
        parts.append(f'<line x1="{ox}" y1="{py(lvl):.1f}" x2="{ox+ow}" y2="{py(lvl):.1f}" '
                     f'stroke="#30363d" stroke-width="1" stroke-dasharray="3 3"/>')
    # spline (overshoots)
    sp = " ".join(f"{px(xs[i]):.1f},{py(sv[i]):.1f}" for i in range(len(xs)))
    parts.append(f'<polyline points="{sp}" fill="none" stroke="#ff6b6b" stroke-width="1.8"/>')
    # pchip
    pp = " ".join(f"{px(xs[i]):.1f},{py(pv[i]):.1f}" for i in range(len(xs)))
    parts.append(f'<polyline points="{pp}" fill="none" stroke="#06d6a0" stroke-width="1.8"/>')
    # data points
    for xi, yi in zip(x, y):
        parts.append(f'<circle cx="{px(xi):.1f}" cy="{py(yi):.1f}" r="4" fill="#ffd43b"/>')
    parts.append(f'<text x="{ox+ow-140}" y="{oy+14}" fill="#ff6b6b" font-size="10">cubic spline</text>')
    parts.append(f'<text x="{ox+ow-140}" y="{oy+29}" fill="#06d6a0" font-size="10">PCHIP</text>')
    parts.append(f'<text x="{ox+ow-140}" y="{oy+44}" fill="#ffd43b" font-size="10">data</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
