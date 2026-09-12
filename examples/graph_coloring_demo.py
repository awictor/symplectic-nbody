"""Demo: graph coloring -- conflict-free scheduling with the fewest colors.

Colors an exam-conflict graph (courses sharing students can't share a slot) with greedy and DSATUR,
finds the exact chromatic number, and draws the graph with each vertex tinted by its color.

    python examples/graph_coloring_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graph_coloring import (greedy_coloring, dsatur_coloring, chromatic_number,  # noqa: E402
                            num_colors, is_proper, brute_chromatic_number)

COURSES = ["Math", "Phys", "Chem", "Bio", "CS", "Econ", "Art"]
# two courses conflict if a student takes both (can't be scheduled together)
CONFLICTS = [
    (0, 1), (0, 4), (1, 2), (1, 4), (2, 3), (3, 5), (4, 5), (0, 2), (2, 4),
]
N = 7


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Graph coloring: scheduling exams so no student has two at once\n")
    print(f"  {N} courses, {len(CONFLICTS)} conflicts (shared students)\n")

    g = greedy_coloring(N, CONFLICTS)
    d = dsatur_coloring(N, CONFLICTS)
    chi, witness = chromatic_number(N, CONFLICTS)

    print(f"  greedy coloring:  {num_colors(g)} time slots  (proper: {is_proper(N, CONFLICTS, g)})")
    print(f"  DSATUR coloring:  {num_colors(d)} time slots  (proper: {is_proper(N, CONFLICTS, d)})")
    print(f"  exact chromatic number: {chi} time slots (the provably minimum)\n")
    print(f"  verified against brute force: {chi == brute_chromatic_number(N, CONFLICTS)}\n")

    slots = {}
    for v in range(N):
        slots.setdefault(witness[v], []).append(COURSES[v])
    print("  optimal exam schedule:")
    for slot in sorted(slots):
        print(f"    slot {slot + 1}: {', '.join(slots[slot])}")

    print("\n  DSATUR colors the most-constrained course next (the one adjacent to the most distinct")
    print("  slots already used), which finds the optimum here where a naive greedy order might not.")
    print("  Deciding k-colorability is NP-complete for k>=3, so the exact number uses branch-and-")
    print("  bound with a clique lower bound and a DSATUR upper bound to prune.")

    _svg(os.path.join(outdir, "graph_coloring.svg"), witness, chi)
    print(f"\n  wrote {os.path.join(outdir, 'graph_coloring.svg')}")


_COLORS = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#ff6b6b", "#b197fc", "#e6edf3"]


def _svg(path, coloring, chi, width=720, height=430):
    cx, cy, r = width / 2, height / 2 + 15, 150
    pos = {}
    for v in range(N):
        ang = 2 * math.pi * v / N - math.pi / 2
        pos[v] = (cx + r * math.cos(ang), cy + r * math.sin(ang))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Exam scheduling by graph coloring: {chi} time slots, no conflict shares one</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'each node is a course tinted by its assigned slot; edges join courses that clash</text>',
    ]

    for u, v in CONFLICTS:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="#484f58" stroke-width="1.5"/>')

    for v, (x, y) in pos.items():
        col = _COLORS[coloring[v] % len(_COLORS)]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="26" fill="{col}"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+4:.1f}" fill="#0d1117" font-size="12" '
                     f'text-anchor="middle" font-weight="bold">{COURSES[v]}</text>')
        parts.append(f'<text x="{x:.1f}" y="{y+18:.1f}" fill="#0d1117" font-size="9" '
                     f'text-anchor="middle">slot {coloring[v]+1}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
