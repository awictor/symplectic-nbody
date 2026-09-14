"""B-spline demo: a cubic curve over its control polygon, plus the Cox-de Boor basis functions (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import b_spline as B


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
YELLOW = "#ffd43b"
GREEN = "#06d6a0"
RED = "#ff6b6b"
PALETTE = ["#4dabf7", "#06d6a0", "#b197fc", "#ff922b", "#ff6b6b", "#ffd43b",
           "#63e6be", "#faa2c1", "#74c0fc"]


def _curve_panel(s, ctrl, p, kn, ox, oy, w, h):
    xs = [c[0] for c in ctrl]
    ys = [c[1] for c in ctrl]
    lo_x, hi_x = min(xs) - 0.5, max(xs) + 0.5
    lo_y, hi_y = min(ys) - 0.5, max(ys) + 0.5

    def sx(x):
        return ox + (x - lo_x) / (hi_x - lo_x) * w

    def sy(y):
        return oy + h - (y - lo_y) / (hi_y - lo_y) * h

    s.append(f'<text x="{ox}" y="{oy-8}" fill="{TEXT}" font-size="13">'
             f'cubic B-spline curve + control polygon</text>')
    # control polygon
    poly = " ".join(f"{sx(c[0]):.1f},{sy(c[1]):.1f}" for c in ctrl)
    s.append(f'<polyline points="{poly}" fill="none" stroke="{GRAY}" '
             f'stroke-width="1" stroke-dasharray="4,3"/>')
    # control points
    for c in ctrl:
        s.append(f'<circle cx="{sx(c[0]):.1f}" cy="{sy(c[1]):.1f}" r="4" fill="{YELLOW}"/>')
    # the curve
    pts = B.sample(ctrl, p, kn, 200)
    cv = " ".join(f"{sx(pt[0]):.1f},{sy(pt[1]):.1f}" for pt in pts)
    s.append(f'<polyline points="{cv}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>')
    # endpoints (clamped: interpolated)
    for c in (ctrl[0], ctrl[-1]):
        s.append(f'<circle cx="{sx(c[0]):.1f}" cy="{sy(c[1]):.1f}" r="5" fill="none" '
                 f'stroke="{GREEN}" stroke-width="2"/>')


def _basis_panel(s, ctrl, p, kn, ox, oy, w, h):
    n_ctrl = len(ctrl)
    u0, u1 = kn[p], kn[n_ctrl]
    s.append(f'<text x="{ox}" y="{oy-8}" fill="{TEXT}" font-size="13">'
             f'Cox-de Boor basis functions N(u) (sum to 1)</text>')
    s.append(f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" fill="none" stroke="{GRAY}" '
             f'stroke-width="0.6"/>')

    def sx(u):
        return ox + (u - u0) / (u1 - u0) * w

    def sy(v):
        return oy + h - v * h

    # gridline at N=1
    s.append(f'<line x1="{ox}" y1="{sy(1):.1f}" x2="{ox+w}" y2="{sy(1):.1f}" stroke="#21262d"/>')
    s.append(f'<text x="{ox-4}" y="{sy(1)+4:.1f}" fill="{GRAY}" font-size="9" '
             f'text-anchor="end">1</text>')
    NS = 200
    for i in range(n_ctrl):
        col = PALETTE[i % len(PALETTE)]
        path = []
        for k in range(NS + 1):
            u = u0 + (u1 - u0) * k / NS
            v = B.basis(i, p, u, kn)
            path.append(f"{sx(u):.1f},{sy(v):.1f}")
        s.append(f'<polyline points="{" ".join(path)}" fill="none" stroke="{col}" '
                 f'stroke-width="1.6"/>')
    # knot ticks
    for kv in sorted(set(kn)):
        if u0 <= kv <= u1:
            s.append(f'<line x1="{sx(kv):.1f}" y1="{oy+h}" x2="{sx(kv):.1f}" y2="{oy+h+5}" '
                     f'stroke="{GRAY}"/>')


def _svg(path, ctrl, p, kn):
    W, H = 760, 340
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="monospace">']
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    _curve_panel(s, ctrl, p, kn, 40, 40, 320, 240)
    _basis_panel(s, ctrl, p, kn, 420, 40, 300, 240)
    s.append(f'<text x="40" y="{H-14}" fill="{GRAY}" font-size="10">'
             f'Each control point (yellow) drives only a local window of the curve; the clamped '
             f'ends (green rings) are interpolated exactly. Basis colors match control indices.</text>')
    s.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(s))


def main(outdir=None):
    ctrl = [[0, 0], [1, 3], [3, 4], [5, 1], [6, 3], [8, 2], [9, 0]]
    p = 3
    kn = B.clamped_knots(len(ctrl), p)

    lines = []
    lines.append("Cubic B-spline curve via the Cox-de Boor recursion")
    lines.append("=" * 54)
    lines.append(f"control points: {len(ctrl)}   degree: {p}   pieces: {len(ctrl) - p}")
    lines.append(f"clamped knot vector: [{', '.join(f'{k:.3f}' for k in kn)}]")
    lines.append("")
    lines.append("partition of unity check (basis sums to 1):")
    lines.append(f"{'u':>8}{'sum N_i(u)':>14}{'#active':>10}")
    for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        u = kn[p] + (kn[len(ctrl)] - kn[p]) * frac
        N = B.all_basis(p, u, kn, len(ctrl))
        active = sum(1 for v in N if v > 1e-9)
        lines.append(f"{u:>8.3f}{sum(N):>14.9f}{active:>10}")
    lines.append("At most p+1 = 4 basis functions are nonzero anywhere -> local control.")
    lines.append("")
    lines.append("endpoint interpolation (clamped):")
    lines.append(f"  C(u_start) = {B.evaluate(ctrl, p, kn[p], kn)}  == P0 {ctrl[0]}")
    lines.append(f"  C(u_end)   = {B.evaluate(ctrl, p, kn[len(ctrl)], kn)}  == P_last {ctrl[-1]}")
    lines.append("")
    lines.append("de Boor vs direct basis summation (max coordinate diff over 100 samples):")
    md = 0.0
    for k in range(100):
        u = kn[p] + (kn[len(ctrl)] - kn[p]) * k / 99
        e = B.evaluate_basis(ctrl, p, u, kn)
        d = B.evaluate(ctrl, p, u, kn)
        md = max(md, max(abs(e[c] - d[c]) for c in range(2)))
    lines.append(f"  {md:.2e}  (the two evaluators agree to machine precision)")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        _svg(os.path.join(outdir, "b_spline.svg"), ctrl, p, kn)

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
