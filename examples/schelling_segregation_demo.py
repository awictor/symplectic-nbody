"""Schelling demo: mild tolerance producing sharp segregation, before/after grids and the tau sweep."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import schelling_segregation as ss


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("Schelling segregation -- mild preference, total segregation (emergence)")
    lines.append("=" * 70)
    lines.append("")

    tau = 1.0 / 3
    res = ss.simulate(40, 40, empty_frac=0.1, tau=tau, seed=1, track=True)
    lines.append(f"40x40 grid, 10% empty, tolerance tau={tau:.2f} (each agent wants only 1/3 same-type).")
    lines.append(f"  initial similarity: {res['initial_similarity']:.3f}")
    lines.append(f"  final similarity:   {res['mean_similarity']:.3f}  (segregation, from a mild wish!)")
    lines.append(f"  converged in {res['rounds']} rounds, {res['happy_fraction']*100:.0f}% happy.")
    lines.append("")
    lines.append("Nobody wanted segregation -- everyone was content with a 1/3 minority -- yet blocks form.")
    lines.append("")

    # segregation vs tolerance
    lines.append("Equilibrium segregation vs tolerance threshold tau:")
    lines.append("   tau     final similarity   happy%")
    lines.append("   " + "-" * 40)
    for t in (0.0, 0.2, 0.33, 0.5, 0.6, 0.75):
        r = ss.simulate(40, 40, 0.1, t, seed=2)
        lines.append(f"   {t:4.2f}    {r['mean_similarity']:.3f}              {r['happy_fraction']*100:.0f}%")
    lines.append("")
    lines.append("Higher demands -> more segregation; but even the tiny tau=0.33 lands at ~0.75 similarity.")
    lines.append("")

    # trajectory
    lines.append("Segregation climbs each round (tau=0.4):")
    r = ss.simulate(40, 40, 0.1, 0.4, seed=1, track=True)
    lines.append("   round   similarity")
    lines.append("   " + "-" * 22)
    for i in range(0, min(len(r["history"]), 8)):
        lines.append(f"   {i:4d}    {r['history'][i][0]:.3f}")

    text = "\n".join(lines)
    print(text)

    svg = _svg(tau)
    return text, svg


def _svg(tau):
    W, H = 640, 400
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Schelling: random mix (left) segregates (right), tau=1/3</text>')

    rows = cols = 50
    grid, rng = ss.make_grid(rows, cols, 0.1, seed=1)
    initial = [row[:] for row in grid]
    sim0 = ss.mean_similarity(grid)
    # run to equilibrium
    for _ in range(60):
        if ss.step(grid, tau, rng) == 0:
            break
    sim1 = ss.mean_similarity(grid)

    cell = 4.2
    panel = cols * cell

    def draw(g, x0, label, sim):
        for r in range(rows):
            for c in range(cols):
                v = g[r][c]
                col = "#0d1117" if v == ss.EMPTY else (P["blue"] if v == 1 else P["red"])
                if v != ss.EMPTY:
                    parts.append(f'<rect x="{x0 + c*cell:.1f}" y="{60 + r*cell:.1f}" '
                                 f'width="{cell:.1f}" height="{cell:.1f}" fill="{col}"/>')
        parts.append(f'<rect x="{x0}" y="{60}" width="{panel}" height="{panel}" fill="none" '
                     f'stroke="{P["gray"]}" stroke-width="1"/>')
        parts.append(f'<text x="{x0}" y="{54}" fill="{P["text"]}" font-size="11">{label}</text>')
        parts.append(f'<text x="{x0}" y="{60 + panel + 14:.1f}" fill="{P["gray"]}" '
                     f'font-size="10">similarity {sim:.2f}</text>')

    draw(initial, 40, "random start", sim0)
    draw(grid, 40 + panel + 40, "equilibrium", sim1)

    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'blue and red agents want only 1/3 like-neighbours, yet self-sort into big single-colour '
                 f'blocks -- macro from micro.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
