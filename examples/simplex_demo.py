"""Demo: the simplex method solving linear programs, with the feasible polytope drawn.

Solves a small 2-D production LP, shows the optimum vertex, confirms LP duality, and draws the
feasible region with the objective's improving direction and the optimal vertex.

    python examples/simplex_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simplex import solve  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("The simplex method: linear programming by walking polytope vertices\n")

    # a small factory: maximize profit 3x + 5y subject to resource limits
    c = [3, 5]
    cons = [([1, 0], "<=", 4),      # machine A hours
            ([0, 2], "<=", 12),     # machine B hours
            ([3, 2], "<=", 18)]     # labour hours
    r = solve(c, cons, maximize=True)

    print("  Factory problem: maximize profit 3x + 5y")
    print("    subject to  x <= 4,  2y <= 12,  3x + 2y <= 18,  x,y >= 0")
    print(f"    -> status {r.status}, optimal profit {r.value:.1f} at "
          f"x = {r.x[0]:.2f}, y = {r.x[1]:.2f}\n")

    # LP duality
    A = [[1, 0], [0, 2], [3, 2]]
    b = [4, 12, 18]
    AT = [[A[i][j] for i in range(len(A))] for j in range(len(c))]
    dual = solve(b, [(AT[j], ">=", c[j]) for j in range(len(c))], maximize=False)
    print(f"  Dual problem (minimize resource prices b.y): optimum {dual.value:.1f}")
    print(f"    LP duality: primal {r.value:.1f} == dual {dual.value:.1f}  "
          f"-> {abs(r.value - dual.value) < 1e-6}\n")

    # a minimization: diet problem
    dr = solve([2, 3], [([1, 1], ">=", 10), ([1, 3], ">=", 18)], maximize=False)
    print("  Diet problem: minimize cost 2x + 3y s.t. x+y>=10, x+3y>=18")
    print(f"    -> minimum cost {dr.value:.1f} at x = {dr.x[0]:.2f}, y = {dr.x[1]:.2f}")

    # unbounded + infeasible detection
    ub = solve([1, 1], [([1, -1], "<=", 1)], maximize=True)
    inf = solve([1], [([1], ">=", 5), ([1], "<=", 2)], maximize=True)
    print(f"\n  Detects unbounded LPs: {ub.status}")
    print(f"  Detects infeasible LPs: {inf.status}")

    print("\n  Simplex starts at a vertex of the feasible polytope and slides to an adjacent vertex")
    print("  that improves the objective, until none does -- then that vertex is provably optimal.")
    print("  Bland's rule (smallest-index pivot) prevents cycling on degenerate problems.")

    _svg(os.path.join(outdir, "simplex.svg"), c, cons, r)
    print(f"\n  wrote {os.path.join(outdir, 'simplex.svg')}")


def _svg(path, c, cons, result, width=760, height=440):
    # draw the 2-D feasible region by sampling a grid and the constraint lines
    scale = 34
    ox, oy = 70, height - 60          # origin in pixels

    def px(x):
        return ox + x * scale

    def py(y):
        return oy - y * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Simplex: feasible polytope and the optimal vertex</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'maximize {c[0]}x + {c[1]}y; green = feasible region, red = optimum, arrows = objective gradient</text>',
    ]

    # feasible region as a filled polygon: sample its boundary by scanning
    def feasible(x, y):
        if x < -1e-9 or y < -1e-9:
            return False
        for a, sense, b in cons:
            lhs = a[0] * x + a[1] * y
            if sense == "<=" and lhs > b + 1e-9:
                return False
            if sense == ">=" and lhs < b - 1e-9:
                return False
        return True

    # build the feasible polygon via its vertices (intersections) then order by angle
    import itertools
    lines = [([1, 0], 0.0), ([0, 1], 0.0)]     # axes
    for a, sense, b in cons:
        lines.append((a, b))
    verts = []
    for (a1, b1), (a2, b2) in itertools.combinations(lines, 2):
        det = a1[0] * a2[1] - a1[1] * a2[0]
        if abs(det) < 1e-12:
            continue
        x = (b1 * a2[1] - a1[1] * b2) / det
        y = (a1[0] * b2 - b1 * a2[0]) / det
        if feasible(x, y):
            verts.append((x, y))
    # dedup + order CCW around centroid
    uniq = []
    for v in verts:
        if not any(abs(v[0] - u[0]) < 1e-6 and abs(v[1] - u[1]) < 1e-6 for u in uniq):
            uniq.append(v)
    if uniq:
        cx = sum(v[0] for v in uniq) / len(uniq)
        cy = sum(v[1] for v in uniq) / len(uniq)
        import math
        uniq.sort(key=lambda v: math.atan2(v[1] - cy, v[0] - cx))
        poly = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in uniq)
        parts.append(f'<polygon points="{poly}" fill="#06d6a0" fill-opacity="0.18" '
                     f'stroke="#06d6a0" stroke-width="2"/>')

    # constraint lines
    for a, sense, b in cons:
        if abs(a[1]) > 1e-9:
            x0, x1 = 0, 8
            y0 = (b - a[0] * x0) / a[1]
            y1 = (b - a[0] * x1) / a[1]
        else:
            x0 = x1 = b / a[0]
            y0, y1 = 0, 10
        parts.append(f'<line x1="{px(x0):.1f}" y1="{py(y0):.1f}" x2="{px(x1):.1f}" y2="{py(y1):.1f}" '
                     f'stroke="#8b949e" stroke-width="1" stroke-dasharray="4,3"/>')

    # axes
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{width-30}" y2="{oy}" stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="40" stroke="#484f58" stroke-width="1.5"/>')

    # objective gradient arrows
    import math
    g = math.hypot(c[0], c[1])
    gx, gy = c[0] / g, c[1] / g
    for base in [(1, 1), (2, 3)]:
        parts.append(f'<line x1="{px(base[0]):.1f}" y1="{py(base[1]):.1f}" '
                     f'x2="{px(base[0] + gx):.1f}" y2="{py(base[1] + gy):.1f}" '
                     f'stroke="#ffd43b" stroke-width="1.8"/>')

    # optimal vertex
    ox_, oy_ = result.x[0], result.x[1]
    parts.append(f'<circle cx="{px(ox_):.1f}" cy="{py(oy_):.1f}" r="7" fill="#ff6b6b"/>')
    parts.append(f'<text x="{px(ox_) + 10:.1f}" y="{py(oy_) - 8:.1f}" fill="#ff6b6b" font-size="13">'
                 f'optimum ({ox_:.0f},{oy_:.0f}) = {result.value:.0f}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
