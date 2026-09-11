"""Demo: Parrondo's paradox -- two losing games that together win.

Prints each game's long-run drift (A and B both lose, the mixture wins) against a Monte-Carlo
run, then draws the capital trajectories of A, B, and the random mixture, and the combined
drift as a function of how often game A is played.

    python examples/parrondo_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parrondo import (game_a_drift, game_b_drift, mixed_drift,  # noqa: E402
                      mixed_winprobs, drift, simulate, simulate_trajectory)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Parrondo's paradox: game A loses, game B loses, but A-and-B-mixed WINS\n")
    print(f"  {'game':>10}{'drift/round':>14}{'sim drift':>12}")
    for name, d, g in (("A alone", game_a_drift(), "A"),
                       ("B alone", game_b_drift(), "B"),
                       ("50/50 mix", mixed_drift(), "mix")):
        print(f"  {name:>10}{d:>14.5f}{simulate(g, rounds=200000, seed=7):>12.5f}")

    print("\n  Game A is a slightly-losing flat coin. Game B flips a terrible coin whenever your")
    print("  capital is a multiple of 3 and a good one otherwise -- and loses because the walk")
    print("  gets stuck visiting the bad state too often. Mixing in game A reshuffles that")
    print("  occupancy so the good coin comes up more, and the combined drift turns positive.")
    print("  The same flashing-ratchet trick drives molecular motors: order pumped from noise.")

    _svg(os.path.join(outdir, "parrondo.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'parrondo.svg')}")


def _svg(path, w=760, h=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Parrondo\'s paradox: losing + losing = winning</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'capital over time for A, B, and the mix (left); combined drift vs mixing fraction (right)</text>',
    ]

    # left: capital trajectories
    rounds = 100000
    every = 500
    trajA = simulate_trajectory("A", rounds, seed=7, every=every)
    trajB = simulate_trajectory("B", rounds, seed=7, every=every)
    trajM = simulate_trajectory("mix", rounds, seed=7, every=every)

    lx0, lx1 = 55, w // 2 - 20
    ly0, ly1 = h - 50, 60
    allv = trajA + trajB + trajM
    vmin, vmax = min(allv), max(allv)
    span = (vmax - vmin) or 1
    npts = len(trajM)

    def LX(k):
        return lx0 + k / (npts - 1) * (lx1 - lx0)

    def LY(v):
        return ly0 - (v - vmin) / span * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    # zero line
    if vmin < 0 < vmax:
        parts.append(f'<line x1="{lx0}" y1="{LY(0):.1f}" x2="{lx1}" y2="{LY(0):.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
    for traj, col, lab in ((trajA, "#ff922b", "A (loses)"),
                           (trajB, "#ff6b6b", "B (loses)"),
                           (trajM, "#06d6a0", "mix (wins)")):
        pts = " ".join(f"{LX(k):.1f},{LY(v):.1f}" for k, v in enumerate(traj))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">rounds ({rounds//1000}k) -> capital</text>')
    ly = ly1
    for col, lab in (("#06d6a0", "mix (wins)"), ("#ff922b", "A (loses)"), ("#ff6b6b", "B (loses)")):
        parts.append(f'<rect x="{lx0+8}" y="{ly}" width="9" height="9" fill="{col}"/>'
                     f'<text x="{lx0+21}" y="{ly+8}" fill="#e6edf3" font-size="9">{lab}</text>')
        ly += 14

    # right: drift vs mixing fraction gamma (fraction of rounds that play game A)
    rx0, rx1 = w // 2 + 45, w - 30
    ry0, ry1 = h - 50, 60
    gammas = [i / 40 for i in range(41)]
    drifts = [mixed_drift(gamma=g) for g in gammas]
    dmin, dmax = min(drifts), max(drifts)
    dspan = (dmax - dmin) or 1

    def RX(g):
        return rx0 + g * (rx1 - rx0)

    def RY(d):
        return ry0 - (d - dmin) / dspan * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    # zero line: separates winning (above) from losing (below)
    parts.append(f'<line x1="{rx0}" y1="{RY(0):.1f}" x2="{rx1}" y2="{RY(0):.1f}" '
                 f'stroke="#8b949e" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{rx1-2:.1f}" y="{RY(0)-4:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">break even</text>')
    pts = " ".join(f"{RX(g):.1f},{RY(d):.1f}" for g, d in zip(gammas, drifts))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2.5"/>')
    # endpoints: gamma=0 is pure B (losing), gamma=1 is pure A (losing)
    parts.append(f'<circle cx="{RX(0):.1f}" cy="{RY(drifts[0]):.1f}" r="3" fill="#ff6b6b"/>')
    parts.append(f'<text x="{RX(0)+4:.1f}" y="{RY(drifts[0])+12:.1f}" fill="#ff6b6b" font-size="9">pure B</text>')
    parts.append(f'<circle cx="{RX(1):.1f}" cy="{RY(drifts[-1]):.1f}" r="3" fill="#ff922b"/>')
    parts.append(f'<text x="{RX(1)-4:.1f}" y="{RY(drifts[-1])+12:.1f}" fill="#ff922b" font-size="9" '
                 f'text-anchor="end">pure A</text>')
    # peak
    kbest = max(range(len(drifts)), key=lambda i: drifts[i])
    parts.append(f'<circle cx="{RX(gammas[kbest]):.1f}" cy="{RY(drifts[kbest]):.1f}" r="3.5" fill="#06d6a0"/>')
    parts.append(f'<text x="{RX(gammas[kbest]):.1f}" y="{RY(drifts[kbest])-6:.1f}" fill="#06d6a0" '
                 f'font-size="9" text-anchor="middle">best mix</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">fraction playing game A</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
