"""Demo: integer programming by branch and bound -- the LP relaxation, the integrality gap, and pruning.

Solves a small production-planning ILP and a 0/1 knapsack, showing the LP relaxation's fractional
optimum, the integer optimum found by branch and bound, the integrality gap between them, and how many
tree nodes the bound pruning saved versus brute force. Draws the LP-vs-ILP objective comparison.

    python examples/integer_programming_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from integer_programming import (  # noqa: E402
    solve_ilp, lp_relaxation_bound, solve_knapsack_ilp, brute_ilp,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Integer programming: branch and bound over the LP relaxation\n")

    # production planning: maximize profit 5x + 4y with resource limits, integer units
    c = [5, 4]
    cons = [([6, 4], "<=", 24), ([1, 2], "<=", 6)]
    lp = lp_relaxation_bound(c, cons, maximize=True)
    ilp = solve_ilp(c, cons, maximize=True, var_bounds={0: (0, 10), 1: (0, 10)})
    print(f"  Production planning: max 5x + 4y  s.t. 6x+4y<=24, x+2y<=6")
    print(f"    LP relaxation optimum: {lp:.2f}  (fractional, not buildable)")
    print(f"    integer optimum:       {ilp['value']:.0f}  at x = {ilp['x']}")
    print(f"    integrality gap:       {lp - ilp['value']:.2f}")
    print(f"    branch-and-bound nodes explored: {ilp['nodes']}\n")

    # 0/1 knapsack
    values = [60, 100, 120, 80, 40]
    weights = [10, 20, 30, 15, 8]
    capacity = 50
    res = solve_knapsack_ilp(values, weights, capacity)
    chosen = [i for i in range(len(values)) if res["x"][i] == 1]
    print(f"  0/1 knapsack: {len(values)} items, capacity {capacity}")
    print(f"    values  {values}")
    print(f"    weights {weights}")
    print(f"    optimal value {res['value']:.0f}, items chosen {chosen} "
          f"(weight {sum(weights[i] for i in chosen)})")

    # brute-force cross-check + node savings
    cons_k = [(list(weights), "<=", capacity)]
    bval, bx = brute_ilp(values, cons_k, True, [(0, 1)] * len(values))
    brute_evals = 2 ** len(values)
    print(f"    brute force checks all {brute_evals} subsets -> value {bval:.0f} (matches: "
          f"{abs(bval - res['value']) < 1e-6})")
    print(f"    branch and bound explored {res['nodes']} LP nodes instead\n")

    print(f"  The LP relaxation gives a bound that lets branch and bound prune whole subtrees:")
    print(f"  once a subproblem's relaxation cannot beat the best integer solution so far, it")
    print(f"  is discarded unexplored. That is what makes exact integer optimization tractable.")

    _svg(os.path.join(outdir, "integer_programming.svg"), lp, ilp["value"], ilp["x"], cons)
    print(f"\n  wrote {os.path.join(outdir, 'integer_programming.svg')}")


def _svg(path, lp_val, ilp_val, ilp_x, cons, width=760, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Feasible integer points, LP relaxation optimum, and the integer optimum</text>',
    ]
    ox, oy, ow, oh = 60, 50, width - 120, height - 100
    xmax, ymax = 6, 6

    def sx(x):
        return ox + ow * x / xmax

    def sy(y):
        return oy + oh * (1 - y / ymax)

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    # feasible integer lattice points
    for x in range(xmax + 1):
        for y in range(ymax + 1):
            if 6 * x + 4 * y <= 24 and x + 2 * y <= 6:
                obj = 5 * x + 4 * y
                # shade by objective
                t = obj / 24.0
                col = f"rgb({int(60+120*t)},{int(80+100*t)},{int(180-60*t)})"
                parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{col}"/>')
    # LP optimum (3, 1.5)
    parts.append(f'<circle cx="{sx(3):.1f}" cy="{sy(1.5):.1f}" r="6" fill="none" '
                 f'stroke="#ff922b" stroke-width="2"/>')
    parts.append(f'<text x="{sx(3)+9:.0f}" y="{sy(1.5):.0f}" fill="#ff922b" font-size="10">'
                 f'LP opt (3, 1.5) = {lp_val:.0f}</text>')
    # ILP optimum
    parts.append(f'<circle cx="{sx(ilp_x[0]):.1f}" cy="{sy(ilp_x[1]):.1f}" r="7" fill="none" '
                 f'stroke="#06d6a0" stroke-width="2.5"/>')
    parts.append(f'<text x="{sx(ilp_x[0])+9:.0f}" y="{sy(ilp_x[1])+14:.0f}" fill="#06d6a0" '
                 f'font-size="10">ILP opt {tuple(ilp_x)} = {ilp_val:.0f}</text>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+22:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">x (dots = feasible integer points, brighter = higher profit)</text>')
    parts.append(f'<text x="{ox-8}" y="{oy+6}" fill="#8b949e" font-size="10" text-anchor="end">y</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
