"""Voter model demo: opinions coarsening to consensus, and the fair consensus probability = initial fraction."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import voter_model as vm


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("Voter model -- local imitation drives a population to consensus")
    lines.append("=" * 62)
    lines.append("")

    lines.append("Consensus probability equals the INITIAL +1 fraction (magnetization martingale):")
    lines.append("   initial +1 frac   P(+1 wins) empirical")
    lines.append("   " + "-" * 40)
    for up0 in (0.2, 0.4, 0.5, 0.6, 0.8):
        p = vm.consensus_probability(8, 8, up0, n_runs=600, seed=1)
        lines.append(f"   {up0:5.2f}             {p:.3f}")
    lines.append("")
    lines.append("Start 70% blue and blue wins ~70% of the time -- outcome is FAIR, not majority-take-all.")
    lines.append("")

    # coarsening trajectory
    res = vm.simulate(30, 30, 0.5, seed=3, track_every=500, max_steps=200000)
    hist = res["history"]
    lines.append("Domains coarsen over time (30x30 grid, balanced start):")
    lines.append("   step        magnetization   #domains")
    lines.append("   " + "-" * 40)
    marks = [0, len(hist) // 6, len(hist) // 3, len(hist) // 2, 2 * len(hist) // 3, len(hist) - 1]
    for i in marks:
        if i < len(hist):
            m, d = hist[i]
            lines.append(f"   {(i+1)*500:6d}      {m:+.3f}          {d}")
    lines.append(f"  reached consensus in {res['steps']} steps -> everyone {'+1' if res['final_opinion']==1 else '-1'}")
    lines.append("")
    lines.append("Magnetization wanders (a martingale) while domains merge; on a finite grid one opinion wins.")

    text = "\n".join(lines)
    print(text)

    svg = _svg()
    return text, svg


def _svg():
    W, H = 640, 400
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Voter model coarsening: opinion grid at four times</text>')

    rows = cols = 40
    rng = vm._Rng(7)
    grid = vm.make_grid(rows, cols, 0.5, rng)
    snapshots = [0, 8000, 40000, 160000]
    cell = 3.0
    gap = 20
    panel = cols * cell
    labels = ["start", "8k steps", "40k steps", "160k steps"]

    step_count = 0
    snap_idx = 0
    x_off = 30
    y_off = 55
    for target in snapshots:
        while step_count < target and not vm.is_consensus(grid):
            vm.step(grid, rng)
            step_count += 1
        px0 = x_off + snap_idx * (panel + gap)
        for r in range(rows):
            for c in range(cols):
                col = P["blue"] if grid[r][c] == 1 else "#2d1a1a"
                if grid[r][c] == 1:
                    col = P["blue"]
                else:
                    col = P["red"]
                parts.append(f'<rect x="{px0 + c*cell:.1f}" y="{y_off + r*cell:.1f}" '
                             f'width="{cell:.1f}" height="{cell:.1f}" fill="{col}"/>')
        m = vm.magnetization(grid)
        parts.append(f'<text x="{px0:.1f}" y="{y_off - 6:.1f}" fill="{P["gray"]}" font-size="9">'
                     f'{labels[snap_idx]}</text>')
        parts.append(f'<text x="{px0:.1f}" y="{y_off + panel + 12:.1f}" fill="{P["gray"]}" '
                     f'font-size="9">m={m:+.2f}</text>')
        snap_idx += 1

    parts.append(f'<text x="20" y="{H - 26}" fill="{P["gray"]}" font-size="11">'
                 f'blue = +1, red = -1. Salt-and-pepper noise coarsens into large domains that swallow '
                 f'each other.</text>')
    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'each copy conserves the mean opinion in expectation, so the eventual winner is '
                 f'chosen fairly by initial share.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
