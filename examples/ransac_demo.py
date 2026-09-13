"""Demo: RANSAC -- robust line and circle fitting against heavy outlier contamination.

Generates points on a known line plus 40% random outliers, fits with both RANSAC and ordinary least
squares, and draws the two fits with inliers/outliers coloured. Same for a circle.

    python examples/ransac_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ransac import ransac_line, ransac_circle, ols_line, line_residual  # noqa: E402


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state >> 8

    return nxt


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    rng = _lcg(2024)

    def rnd():
        return rng() / (1 << 24)

    print("RANSAC: consensus fitting that ignores outliers\n")

    # --- line: y = 1.5 x + 2, plus 40% outliers ---
    m_true, b_true = 1.5, 2.0
    pts = []
    for _ in range(60):
        x = rnd() * 10
        pts.append((x, m_true * x + b_true + (rnd() - 0.5) * 0.4))
    for _ in range(40):
        pts.append((rnd() * 10, rnd() * 20))

    res = ransac_line(pts, threshold=0.6, seed=3, max_iterations=500)
    ols = ols_line(pts)
    print(f"  Line fit to 60 inliers + 40 outliers (true y = {m_true}x + {b_true}):")
    print(f"    RANSAC found {len(res['inliers'])} inliers in {res['iterations']} iterations")
    # convert normalized (a,b,c) back to slope/intercept for display
    a, b, c = res["model"]
    if abs(b) > 1e-9:
        print(f"    RANSAC line:  y = {-a/b:.3f} x + {-c/b:.3f}   (matches truth)")
    ao, bo, co = ols
    if abs(bo) > 1e-9:
        print(f"    OLS line:     y = {-ao/bo:.3f} x + {-co/bo:.3f}   (dragged off by outliers)")

    # --- circle ---
    cx0, cy0, r0 = 5.0, 5.0, 4.0
    cpts = []
    for _ in range(70):
        ang = rnd() * 2 * math.pi
        rad = r0 + (rnd() - 0.5) * 0.3
        cpts.append((cx0 + rad * math.cos(ang), cy0 + rad * math.sin(ang)))
    for _ in range(35):
        cpts.append((rnd() * 12 - 1, rnd() * 12 - 1))
    cres = ransac_circle(cpts, threshold=0.5, seed=9, max_iterations=800)
    cx, cy, r = cres["model"]
    print(f"\n  Circle fit to 70 inliers + 35 outliers (true center ({cx0},{cy0}) r={r0}):")
    print(f"    RANSAC center ({cx:.2f}, {cy:.2f}) r={r:.2f}, {len(cres['inliers'])} inliers")

    print("\n  RANSAC draws a minimal sample, counts agreeing points, keeps the biggest consensus.")
    print("  One clean sample reveals the model, so it survives outlier fractions that wreck OLS.")

    _svg(os.path.join(outdir, "ransac.svg"), pts, res, ols, cpts, cres)
    print(f"\n  wrote {os.path.join(outdir, 'ransac.svg')}")


def _svg(path, pts, res, ols, cpts, cres, width=760, height=430):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'RANSAC (green) vs least squares (red) with 40% outliers; right: robust circle</text>',
    ]

    # ---- left panel: line ---------------------------------------------------------------
    lx0, ly0, lw, lh = 40, 50, 340, 340
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    def sx(x):
        return lx0 + lw * (x - xmin) / (xmax - xmin or 1)

    def sy(y):
        return ly0 + lh * (1 - (y - ymin) / (ymax - ymin or 1))

    inset = set(res["inliers"])
    for i, (x, y) in enumerate(pts):
        col = "#06d6a0" if i in inset else "#8b949e"
        rr = 3 if i in inset else 2
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{rr}" fill="{col}" '
                     f'opacity="0.8"/>')

    def draw_line(model, col):
        a, b, c = model
        # sample two x's, solve for y (b != 0 assumed here)
        if abs(b) < 1e-9:
            return
        pts2 = [(xmin, -(a * xmin + c) / b), (xmax, -(a * xmax + c) / b)]
        parts.append(f'<line x1="{sx(pts2[0][0]):.1f}" y1="{sy(pts2[0][1]):.1f}" '
                     f'x2="{sx(pts2[1][0]):.1f}" y2="{sy(pts2[1][1]):.1f}" '
                     f'stroke="{col}" stroke-width="2.5"/>')

    draw_line(ols, "#ff6b6b")
    draw_line(res["model"], "#06d6a0")

    # ---- right panel: circle ------------------------------------------------------------
    ox0, oy0, ow, oh = 420, 50, 300, 340
    cxs = [p[0] for p in cpts]
    cys = [p[1] for p in cpts]
    cxmin, cxmax = min(cxs), max(cxs)
    cymin, cymax = min(cys), max(cys)
    scale = min(ow / (cxmax - cxmin or 1), oh / (cymax - cymin or 1))

    def cx_(x):
        return ox0 + (x - cxmin) * scale

    def cy_(y):
        return oy0 + oh - (y - cymin) * scale

    cinset = set(cres["inliers"])
    for i, (x, y) in enumerate(cpts):
        col = "#4dabf7" if i in cinset else "#8b949e"
        rr = 3 if i in cinset else 2
        parts.append(f'<circle cx="{cx_(x):.1f}" cy="{cy_(y):.1f}" r="{rr}" fill="{col}" '
                     f'opacity="0.8"/>')
    ccx, ccy, cr = cres["model"]
    parts.append(f'<circle cx="{cx_(ccx):.1f}" cy="{cy_(ccy):.1f}" r="{cr*scale:.1f}" '
                 f'fill="none" stroke="#ffd43b" stroke-width="2.5"/>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
