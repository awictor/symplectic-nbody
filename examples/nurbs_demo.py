"""NURBS demo: an exact circle no polynomial can draw, plus the weight knob pulling a curve (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import nurbs as NU
import b_spline


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
YELLOW = "#ffd43b"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
RED = "#ff6b6b"
PURPLE = "#b197fc"


def _circle_panel(s, ox, oy, w, h):
    ctrl, wt, p, kn = NU.circle((0, 0), 1.0)
    lo, hi = -1.6, 1.6

    def sx(x):
        return ox + (x - lo) / (hi - lo) * w

    def sy(y):
        return oy + h - (y - lo) / (hi - lo) * h

    s.append(f'<text x="{ox}" y="{oy-8}" fill="{TEXT}" font-size="13">'
             f'exact circle: degree-2 NURBS, 9 points</text>')
    # control polygon
    poly = " ".join(f"{sx(c[0]):.1f},{sy(c[1]):.1f}" for c in ctrl)
    s.append(f'<polyline points="{poly}" fill="none" stroke="{GRAY}" '
             f'stroke-width="0.8" stroke-dasharray="3,3"/>')
    # the exact NURBS circle
    pts = NU.sample(ctrl, wt, p, kn, 200)
    cv = " ".join(f"{sx(pt[0]):.1f},{sy(pt[1]):.1f}" for pt in pts)
    s.append(f'<polyline points="{cv}" fill="none" stroke="{GREEN}" stroke-width="2.5"/>')
    # a degree-2 polynomial B-spline through the same 4 axis points, for contrast
    axis = [ctrl[0], ctrl[2], ctrl[4], ctrl[6], ctrl[0], ctrl[2]]
    kb = b_spline.clamped_knots(len(axis), 2)
    bpts = b_spline.sample(axis, 2, kb, 120)
    bcv = " ".join(f"{sx(pt[0]):.1f},{sy(pt[1]):.1f}" for pt in bpts)
    s.append(f'<polyline points="{bcv}" fill="none" stroke="{RED}" '
             f'stroke-width="1.3" stroke-dasharray="5,3"/>')
    # control points, weights labeled
    for i, c in enumerate(ctrl[:-1]):
        col = YELLOW if wt[i] == 1 else PURPLE
        s.append(f'<circle cx="{sx(c[0]):.1f}" cy="{sy(c[1]):.1f}" r="3.5" fill="{col}"/>')
    s.append(f'<text x="{ox}" y="{oy+h+16}" fill="{GREEN}" font-size="10">green = exact NURBS circle '
             f'(corner weights sqrt2/2)</text>')
    s.append(f'<text x="{ox}" y="{oy+h+30}" fill="{RED}" font-size="10">red dashed = polynomial '
             f'B-spline (cannot be a circle)</text>')


def _weight_panel(s, ox, oy, w, h):
    cc = [[0, 0], [1, 3], [3, 3.2], [5, 0]]
    deg = 3
    kn = b_spline.clamped_knots(len(cc), deg)
    xs = [c[0] for c in cc]
    ys = [c[1] for c in cc]
    lo_x, hi_x = min(xs) - 0.5, max(xs) + 0.5
    lo_y, hi_y = min(ys) - 0.5, max(ys) + 1

    def sx(x):
        return ox + (x - lo_x) / (hi_x - lo_x) * w

    def sy(y):
        return oy + h - (y - lo_y) / (hi_y - lo_y) * h

    s.append(f'<text x="{ox}" y="{oy-8}" fill="{TEXT}" font-size="13">'
             f'weight knob: pulling toward point 2</text>')
    poly = " ".join(f"{sx(c[0]):.1f},{sy(c[1]):.1f}" for c in cc)
    s.append(f'<polyline points="{poly}" fill="none" stroke="{GRAY}" '
             f'stroke-width="0.8" stroke-dasharray="3,3"/>')
    cols = [BLUE, GREEN, YELLOW, RED]
    for wi, wv in enumerate((1.0, 2.0, 5.0, 15.0)):
        weights = [1.0, 1.0, wv, 1.0]
        pts = NU.sample(cc, weights, deg, kn, 120)
        cv = " ".join(f"{sx(pt[0]):.1f},{sy(pt[1]):.1f}" for pt in pts)
        s.append(f'<polyline points="{cv}" fill="none" stroke="{cols[wi]}" stroke-width="1.8"/>')
        s.append(f'<text x="{ox+w-70}" y="{oy+14+wi*15}" fill="{cols[wi]}" font-size="10">'
                 f'w2 = {wv:g}</text>')
    for c in cc:
        s.append(f'<circle cx="{sx(c[0]):.1f}" cy="{sy(c[1]):.1f}" r="3.5" fill="{YELLOW}"/>')


def _svg(path):
    W, H = 760, 380
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="monospace">']
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    _circle_panel(s, 40, 45, 300, 260)
    _weight_panel(s, 420, 45, 300, 260)
    s.append(f'<text x="40" y="{H-14}" fill="{GRAY}" font-size="10">'
             f'The rational weights are the extra freedom polynomials lack: one quadratic NURBS is an '
             f'exact circle, and raising a control weight pulls the curve toward that point.</text>')
    s.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(s))


def main(outdir=None):
    ctrl, w, p, kn = NU.circle((0, 0), 1.0)

    lines = []
    lines.append("NURBS: the curves that draw exact circles")
    lines.append("=" * 50)
    lines.append(f"standard circle: degree {p}, {len(ctrl)} control points")
    lines.append(f"corner weights = cos(45 deg) = sqrt(2)/2 = {math.sqrt(2)/2:.6f}")
    lines.append("")
    lines.append("radius error around the NURBS circle (should be ~0):")
    lines.append(f"{'u':>8}{'x':>10}{'y':>10}{'radius':>12}{'|r-1|':>12}")
    for frac in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75):
        u = frac
        pt = NU.evaluate(ctrl, w, p, u, kn)
        r = math.hypot(pt[0], pt[1])
        lines.append(f"{u:>8.3f}{pt[0]:>10.5f}{pt[1]:>10.5f}{r:>12.9f}{abs(r-1):>12.2e}")
    lines.append("A polynomial curve of any degree CANNOT satisfy r=1 exactly; the rational")
    lines.append("weights are exactly the missing degree of freedom.")
    lines.append("")

    # arc length vs 2 pi
    circ = NU.sample(ctrl, w, p, kn, 2000)
    L = sum(math.dist(circ[i], circ[i + 1]) for i in range(len(circ) - 1))
    lines.append(f"sampled arc length: {L:.6f}   2*pi = {2*math.pi:.6f}   diff {abs(L-2*math.pi):.2e}")
    lines.append("")

    # weight pull
    cc = [[0, 0], [1, 3], [3, 3.2], [5, 0]]
    deg = 3
    knb = b_spline.clamped_knots(len(cc), deg)
    mid = (knb[deg] + knb[len(cc)]) / 2
    lines.append("Weight knob: distance from curve midpoint to control point 2 as its weight rises:")
    lines.append(f"{'weight':>8}{'dist to P2':>14}")
    for wv in (1.0, 2.0, 5.0, 15.0, 50.0):
        weights = [1.0, 1.0, wv, 1.0]
        pt = NU.evaluate(cc, weights, deg, mid, knb)
        lines.append(f"{wv:>8g}{math.dist(pt, cc[2]):>14.4f}")
    lines.append("Higher weight => the curve is pulled toward that control point.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        _svg(os.path.join(outdir, "nurbs.svg"))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
