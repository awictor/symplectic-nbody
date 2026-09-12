"""Demo: Q-learning learning an optimal gridworld policy from experience alone.

Trains Q-learning on a gridworld without ever seeing its transition model, compares the learned
policy to value iteration's ground truth, shows the learning curve, and draws the learned policy and
value function. Also runs SARSA for contrast.

    python examples/q_learning_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from q_learning import GridEnv, q_learning, sarsa, greedy_rollout  # noqa: E402
from mdp import gridworld  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Q-learning: model-free reinforcement learning from experience\n")

    W, H = 6, 5
    goal = (5, 4)
    obstacles = [(1, 1), (2, 1), (3, 3), (4, 2)]

    env = GridEnv(W, H, goal=goal, obstacles=obstacles, slip=0.0, seed=1)
    Q, policy, returns = q_learning(env, episodes=3000, alpha=0.5, gamma=0.9, epsilon=0.2, seed=2)

    # ground truth
    g = gridworld(W, H, goal=goal, obstacles=obstacles, slip=0.0, gamma=0.9)
    V_opt, pol_opt, _ = g.value_iteration()

    non_goal = [s for s in env.states if s != goal]
    matches = sum(1 for s in non_goal if policy[s] == pol_opt[s])
    print(f"  {W}x{H} gridworld, {len(obstacles)} obstacles, goal {goal}")
    print(f"  Q-learning saw only (state, action, reward, next-state) samples -- never the model.")
    print(f"  after 3000 episodes: policy matches value iteration on {matches}/{len(non_goal)} states")

    opt_steps = greedy_rollout(GridEnv(W, H, goal=goal, obstacles=obstacles, seed=1), pol_opt)
    ql_steps = greedy_rollout(GridEnv(W, H, goal=goal, obstacles=obstacles, seed=1), policy)
    print(f"  greedy rollout: Q-learning {ql_steps} steps vs value-iteration optimum {opt_steps}")

    # SARSA
    senv = GridEnv(W, H, goal=goal, obstacles=obstacles, slip=0.0, seed=1)
    _, pol_s, ret_s = sarsa(senv, episodes=3000, alpha=0.5, gamma=0.9, epsilon=0.2, seed=3)
    sarsa_steps = greedy_rollout(GridEnv(W, H, goal=goal, obstacles=obstacles, seed=1), pol_s)
    print(f"  SARSA (on-policy) rollout: {sarsa_steps} steps")

    print("\n  Learned policy (G = goal, # = obstacle):")
    arrow = {"N": "^", "S": "v", "E": ">", "W": "<"}
    for y in range(H - 1, -1, -1):
        row = []
        for x in range(W):
            if (x, y) in obstacles:
                row.append("#")
            elif (x, y) == goal:
                row.append("G")
            else:
                row.append(arrow[policy[(x, y)]])
        print("    " + " ".join(row))

    print("\n  The TD update nudges Q(s,a) toward r + gamma max_a' Q(s',a') after each step -- learning")
    print("  the optimal action-values from raw experience, off-policy, while exploring epsilon-greedily.")

    _svg(os.path.join(outdir, "q_learning.svg"), returns, ret_s, W, H, goal, obstacles, Q, policy)
    print(f"\n  wrote {os.path.join(outdir, 'q_learning.svg')}")


def _svg(path, returns, ret_s, W, H, goal, obstacles, Q, policy, width=760, height=440):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Q-learning: learning curve (left) and learned policy (right)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'left: episode return rising as the agent learns; right: learned value (color) + policy (arrows)</text>',
        '<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" '
        'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#e6edf3"/></marker></defs>',
    ]

    # left: smoothed learning curve
    lx0, ly0, lw, lh = 55, 90, 320, 300
    parts.append(f'<line x1="{lx0}" y1="{ly0+lh}" x2="{lx0+lw}" y2="{ly0+lh}" stroke="#484f58"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly0+lh}" stroke="#484f58"/>')

    def smooth(xs, w=50):
        out = []
        for i in range(len(xs)):
            lo = max(0, i - w)
            out.append(sum(xs[lo:i + 1]) / (i + 1 - lo))
        return out

    sm_q = smooth(returns)
    sm_s = smooth(ret_s)
    allv = sm_q + sm_s
    lo, hi = min(allv), max(allv)
    rng = hi - lo or 1
    n = len(sm_q)

    def qy(v):
        return ly0 + lh - (v - lo) / rng * lh

    step = max(1, n // 400)
    for series, col, name in ((sm_q, "#4dabf7", "Q-learning"), (sm_s, "#06d6a0", "SARSA")):
        pts = " ".join(f"{lx0 + i/n*lw:.1f},{qy(series[i]):.1f}" for i in range(0, n, step))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.8"/>')
    parts.append(f'<text x="{lx0}" y="{ly0-6}" fill="#8b949e" font-size="11">'
                 f'episode return (smoothed): blue Q-learning, green SARSA</text>')

    # right: value + policy grid
    gx0, gy0 = 410, 90
    gs = min((width - gx0 - 30) / W, (height - gy0 - 40) / H)
    vals = [max(Q[s].values()) for s in Q if s not in obstacles]
    vmin, vmax = min(vals), max(vals)
    vr = vmax - vmin or 1
    arrow = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
    for y in range(H):
        for x in range(W):
            px = gx0 + x * gs
            py = gy0 + (H - 1 - y) * gs
            if (x, y) in obstacles:
                parts.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{gs-1.5:.1f}" '
                             f'height="{gs-1.5:.1f}" fill="#161b22"/>')
                continue
            if (x, y) == goal:
                parts.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{gs-1.5:.1f}" '
                             f'height="{gs-1.5:.1f}" fill="#ffd43b"/>')
                continue
            t = (max(Q[(x, y)].values()) - vmin) / vr
            r, gg, b = int(20 + t * 40), int(60 + t * 150), int(90 + t * 120)
            parts.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{gs-1.5:.1f}" height="{gs-1.5:.1f}" '
                         f'fill="rgb({r},{gg},{b})"/>')
            dx, dy = arrow[policy[(x, y)]]
            cxp, cyp = px + gs / 2, py + gs / 2
            parts.append(f'<line x1="{cxp-dx*gs*0.25:.1f}" y1="{cyp-dy*gs*0.25:.1f}" '
                         f'x2="{cxp+dx*gs*0.25:.1f}" y2="{cyp+dy*gs*0.25:.1f}" stroke="#e6edf3" '
                         f'stroke-width="1.8" marker-end="url(#ar)"/>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
