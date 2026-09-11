"""Demo: the Hungarian algorithm solving the assignment problem optimally.

Assigns workers to jobs at minimum total cost, compares to brute force and the greedy heuristic, and
draws the cost matrix with the chosen optimal assignment highlighted.

    python examples/hungarian_demo.py [output_dir]
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hungarian import solve  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("The Hungarian algorithm: optimal assignment in O(n^3)\n")

    workers = ["Ann", "Bob", "Cy", "Dot"]
    jobs = ["cook", "clean", "drive", "shop"]
    # cost[worker][job]: hours each worker takes at each job
    cost = [[9, 11, 14, 11],
            [6, 15, 13, 10],
            [12, 13, 6, 8],
            [11, 9, 10, 12]]

    print("  Cost matrix (hours for each worker to do each job):")
    print("           " + "  ".join(f"{j:>6}" for j in jobs))
    for i, w in enumerate(workers):
        print(f"    {w:>4}  " + "  ".join(f"{cost[i][j]:6d}" for j in range(4)))

    assign, total = solve(cost)
    print(f"\n  Optimal assignment (total {total} hours):")
    for i, w in enumerate(workers):
        print(f"    {w:>4} -> {jobs[assign[i]]:>6} ({cost[i][assign[i]]} h)")

    # brute force confirmation
    brute = min(sum(cost[i][p[i]] for i in range(4)) for p in itertools.permutations(range(4)))
    print(f"\n  Brute force over all 4! = 24 permutations: {brute}  (matches: {brute == total})")

    # greedy heuristic for comparison
    used = set()
    greedy_total = 0
    for i in range(4):
        j = min((jj for jj in range(4) if jj not in used), key=lambda jj: cost[i][jj])
        used.add(j)
        greedy_total += cost[i][j]
    print(f"  Greedy (pick each row's cheapest free job): {greedy_total}  "
          f"-> {greedy_total - total} hours worse than optimal")

    print("\n  The algorithm reduces rows and columns to expose zeros, selects n independent zeros")
    print("  (one per row and column), and when fewer than n exist, covers the zeros with a minimum")
    print("  set of lines and shifts the smallest uncovered value to create new ones -- until an")
    print("  optimal set of independent zeros appears. O(n^3), versus n! for brute force.")

    _svg(os.path.join(outdir, "hungarian.svg"), workers, jobs, cost, assign)
    print(f"\n  wrote {os.path.join(outdir, 'hungarian.svg')}")


def _svg(path, workers, jobs, cost, assign, width=760, height=430):
    n = len(workers)
    cell = 74
    ox, oy = 150, 90

    vmax = max(max(row) for row in cost)
    vmin = min(min(row) for row in cost)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Hungarian algorithm: optimal assignment (green cells)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'cost matrix shaded by value (darker = cheaper); the chosen minimum-cost cells outlined green</text>',
    ]

    # column headers
    for j, job in enumerate(jobs):
        parts.append(f'<text x="{ox + j*cell + cell/2:.0f}" y="{oy - 10:.0f}" fill="#8b949e" '
                     f'font-size="12" text-anchor="middle">{job}</text>')
    for i, w in enumerate(workers):
        parts.append(f'<text x="{ox - 12:.0f}" y="{oy + i*cell + cell/2 + 4:.0f}" fill="#8b949e" '
                     f'font-size="12" text-anchor="end">{w}</text>')

    for i in range(n):
        for j in range(n):
            x = ox + j * cell
            y = oy + i * cell
            # shade: cheaper = darker blue, costlier = lighter
            t = (cost[i][j] - vmin) / (vmax - vmin + 1e-9)
            shade = int(30 + t * 90)
            parts.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{cell-3}" height="{cell-3}" '
                         f'fill="rgb({shade},{shade+20},{shade+50})"/>')
            chosen = assign[i] == j
            if chosen:
                parts.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{cell-3}" height="{cell-3}" '
                             f'fill="none" stroke="#06d6a0" stroke-width="3.5"/>')
            parts.append(f'<text x="{x + (cell-3)/2:.0f}" y="{y + (cell-3)/2 + 5:.0f}" '
                         f'fill="{"#06d6a0" if chosen else "#e6edf3"}" font-size="16" '
                         f'text-anchor="middle" font-weight="{"bold" if chosen else "normal"}">'
                         f'{cost[i][j]}</text>')

    total = sum(cost[i][assign[i]] for i in range(n))
    parts.append(f'<text x="{ox}" y="{oy + n*cell + 28:.0f}" fill="#06d6a0" font-size="14">'
                 f'optimal total cost = {total}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
