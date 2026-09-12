"""Demo: solving a gridworld MDP by value iteration and policy iteration.

Solves a gridworld with obstacles and stochastic slip, prints the optimal policy as arrows and the
value function, confirms value and policy iteration agree, and draws the value heatmap with the
policy overlaid.

    python examples/mdp_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mdp import gridworld  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Markov decision process: value iteration and policy iteration\n")

    W, H = 5, 4
    goal = (4, 3)
    obstacles = [(1, 1), (1, 2), (3, 1)]
    g = gridworld(W, H, goal=goal, obstacles=obstacles, slip=0.1, gamma=0.9,
                  step_reward=-0.04, goal_reward=1.0)

    V_vi, pol_vi, vi_it = g.value_iteration()
    V_pi, pol_pi, pi_it = g.policy_iteration()

    print(f"  {W}x{H} gridworld, goal at {goal}, obstacles {obstacles}, 10% slip:")
    print(f"    value iteration converged in {vi_it} sweeps")
    print(f"    policy iteration converged in {pi_it} rounds")
    print(f"    same policy: {pol_vi == pol_pi}, same values: "
          f"{all(abs(V_vi[s]-V_pi[s]) < 1e-6 for s in g.states)}")
    print(f"    Bellman residual: {g.bellman_residual(V_vi):.2e} (zero = optimal fixed point)\n")

    arrow = {"N": "^", "S": "v", "E": ">", "W": "<", None: "G"}
    print("  Optimal policy (G = goal, # = obstacle):")
    for y in range(H - 1, -1, -1):
        row = []
        for x in range(W):
            if (x, y) in obstacles:
                row.append("#")
            elif (x, y) == goal:
                row.append("G")
            else:
                row.append(arrow[pol_vi[(x, y)]])
        print("    " + " ".join(row))

    print("\n  State values (higher = closer to the goal along the optimal path):")
    for y in range(H - 1, -1, -1):
        row = []
        for x in range(W):
            if (x, y) in obstacles:
                row.append("  ##  ")
            else:
                row.append(f"{V_vi[(x, y)]:+5.2f} ")
        print("    " + " ".join(row))

    print("\n  Value iteration applies the Bellman optimality backup until the value function stops")
    print("  changing (a contraction, converging geometrically). Policy iteration alternates exact")
    print("  policy evaluation with greedy improvement, converging in a handful of rounds. Both find")
    print("  the same optimal policy -- the action in each state that maximizes expected return.")

    _svg(os.path.join(outdir, "mdp.svg"), W, H, goal, obstacles, V_vi, pol_vi)
    print(f"\n  wrote {os.path.join(outdir, 'mdp.svg')}")


def _svg(path, W, H, goal, obstacles, V, policy, width=760, height=430):
    cell = min((width - 80) / W, (height - 100) / H)
    ox = (width - W * cell) / 2
    oy = 80

    vals = [V[s] for s in V if s not in obstacles]
    vmin, vmax = min(vals), max(vals)
    vr = vmax - vmin or 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Gridworld MDP: optimal value (color) and policy (arrows)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'brighter = higher value (closer to goal); arrows = optimal action; gold = goal, dark = obstacle</text>',
    ]
    arrow = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
    for y in range(H):
        for x in range(W):
            px = ox + x * cell
            py = oy + (H - 1 - y) * cell
            if (x, y) in obstacles:
                parts.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{cell-2:.1f}" '
                             f'height="{cell-2:.1f}" fill="#161b22"/>')
                continue
            if (x, y) == goal:
                parts.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{cell-2:.1f}" '
                             f'height="{cell-2:.1f}" fill="#ffd43b"/>')
                parts.append(f'<text x="{px+cell/2:.1f}" y="{py+cell/2+5:.1f}" fill="#0d1117" '
                             f'font-size="16" text-anchor="middle" font-weight="bold">G</text>')
                continue
            t = (V[(x, y)] - vmin) / vr
            r, gg, b = int(20 + t * 40), int(60 + t * 150), int(90 + t * 120)
            parts.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{cell-2:.1f}" height="{cell-2:.1f}" '
                         f'fill="rgb({r},{gg},{b})"/>')
            parts.append(f'<text x="{px+cell/2:.1f}" y="{py+16:.1f}" fill="#e6edf3" font-size="9" '
                         f'text-anchor="middle">{V[(x,y)]:+.2f}</text>')
            a = policy[(x, y)]
            if a:
                dx, dy = arrow[a]
                cxp, cyp = px + cell / 2, py + cell / 2 + 6
                parts.append(f'<line x1="{cxp-dx*12:.1f}" y1="{cyp-dy*12:.1f}" '
                             f'x2="{cxp+dx*12:.1f}" y2="{cyp+dy*12:.1f}" stroke="#e6edf3" '
                             f'stroke-width="2" marker-end="url(#ar)"/>')
    parts.insert(4, '<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" '
                    'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#e6edf3"/></marker></defs>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
