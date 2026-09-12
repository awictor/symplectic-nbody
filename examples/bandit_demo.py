"""Demo: multi-armed bandit policies and the exploration-exploitation tradeoff.

Runs epsilon-greedy, UCB1, and Thompson sampling against a Bernoulli bandit, compares their
cumulative regret to random selection, and shows each finding the best arm. Draws the regret curves.

    python examples/bandit_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bandit import (BernoulliBandit, epsilon_greedy, ucb1, thompson_sampling, random_policy)  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Multi-armed bandit: exploration vs exploitation\n")

    probs = [0.20, 0.50, 0.75, 0.40, 0.60]
    rounds = 3000
    print(f"  {len(probs)} arms with payout probabilities {probs}")
    print(f"  best arm is #{probs.index(max(probs))} (p = {max(probs)}), over {rounds} rounds:\n")

    policies = [
        ("random", lambda b: random_policy(b, rounds)),
        ("epsilon-greedy", lambda b: epsilon_greedy(b, rounds, epsilon=0.1)),
        ("UCB1", lambda b: ucb1(b, rounds)),
        ("Thompson", lambda b: thompson_sampling(b, rounds)),
    ]

    curves = {}
    print(f"    {'policy':>16}  {'total reward':>12}  {'regret':>8}  {'best-arm %':>10}")
    for name, fn in policies:
        b = BernoulliBandit(probs, seed=11)
        rew, reg, counts, means = fn(b)
        curves[name] = reg
        best_pct = 100 * counts[b.best] / rounds
        print(f"    {name:>16}  {sum(rew):>12.0f}  {reg[-1]:>8.1f}  {best_pct:>9.1f}%")

    print("\n  Regret is reward lost versus always pulling the best arm. Random selection accrues")
    print("  regret linearly; UCB1 and Thompson learn which arm is best and their regret flattens")
    print("  (grows only logarithmically), driving the average regret per round toward zero.")

    # optimism-under-uncertainty snapshot for UCB1
    import math
    b = BernoulliBandit(probs, seed=11)
    _, _, counts, means = ucb1(b, 500)
    print("\n  UCB1 after 500 rounds (mean + confidence bonus per arm):")
    for i in range(len(probs)):
        bonus = math.sqrt(2 * math.log(500) / counts[i]) if counts[i] else float("inf")
        print(f"    arm {i}: pulled {counts[i]:3d}, mean {means[i]:.3f}, "
              f"UCB {means[i] + bonus:.3f}  (true {probs[i]})")

    _svg(os.path.join(outdir, "bandit.svg"), curves, rounds)
    print(f"\n  wrote {os.path.join(outdir, 'bandit.svg')}")


def _svg(path, curves, rounds, width=760, height=430):
    m_left, m_bot, m_top, m_right = 60, 55, 80, 130
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    ymax = max(c[-1] for c in curves.values()) * 1.05

    def px(t):
        return m_left + t / rounds * pw

    def py(v):
        return m_top + ph - v / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Multi-armed bandit: cumulative regret by policy</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'lower and flatter is better; learning policies bend away from the linear random baseline</text>',
    ]
    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height-12}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">round</text>')

    colors = {"random": "#ff6b6b", "epsilon-greedy": "#ff922b", "UCB1": "#4dabf7", "Thompson": "#06d6a0"}
    step = max(1, rounds // 400)
    ly = m_top + 6
    for name, reg in curves.items():
        col = colors.get(name, "#8b949e")
        pts = " ".join(f"{px(i):.1f},{py(reg[i]):.1f}" for i in range(0, len(reg), step))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.8"/>')
        parts.append(f'<line x1="{m_left+pw+12}" y1="{ly}" x2="{m_left+pw+30}" y2="{ly}" '
                     f'stroke="{col}" stroke-width="2.5"/>')
        parts.append(f'<text x="{m_left+pw+34}" y="{ly+4}" fill="#e6edf3" font-size="10">{name}</text>')
        ly += 18

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
