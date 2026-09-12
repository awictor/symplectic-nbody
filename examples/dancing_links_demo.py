"""Demo: Dancing Links (DLX) -- Knuth's Algorithm X for exact cover.

Solves Knuth's textbook exact-cover instance, counts N-queens solutions, and tiles a small board with
dominoes -- all as exact-cover problems. Draws one N-queens solution on a board.

    python examples/dancing_links_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dancing_links import solve_exact_cover, n_queens  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Dancing Links (DLX): exact cover by Algorithm X\n")

    items = [1, 2, 3, 4, 5, 6, 7]
    options = [('A', {1, 4, 7}), ('B', {1, 4}), ('C', {4, 5, 7}),
               ('D', {3, 5, 6}), ('E', {2, 3, 6, 7}), ('F', {2, 7})]
    print("  Knuth's example -- cover items 1..7 with disjoint options:")
    for rid, s in options:
        print(f"    {rid}: {sorted(s)}")
    sol = solve_exact_cover(items, options)
    print(f"    exact cover: {sol[0]}  (options {sol[0]} partition 1..7 exactly)\n")

    print("  N-queens as exact cover (place n non-attacking queens):")
    for n in range(1, 9):
        print(f"    {n}-queens: {n_queens(n)} solution(s)")

    # solve one 6-queens placement and show it
    board = _one_queens_solution(6)
    print("\n  one 6-queens solution (Q = queen):")
    for r in range(6):
        print("    " + " ".join("Q" if board[r] == c else "." for c in range(6)))

    print("\n  DLX stores the cover matrix as a toroidal doubly-linked list; 'covering' a column")
    print("  splices it and its conflicting options out with x.L.R=x.R; x.R.L=x.L, and 'uncovering'")
    print("  on backtrack restores them with the exact inverse -- no allocation during the search,")
    print("  and the pointers literally dance out and back. The column-with-fewest-options heuristic")
    print("  keeps the branching minimal.")

    _svg(os.path.join(outdir, "dancing_links.svg"), board)
    print(f"\n  wrote {os.path.join(outdir, 'dancing_links.svg')}")


def _one_queens_solution(n):
    """Recover one N-queens placement as a list: row -> column."""
    items = [f"R{i}" for i in range(n)] + [f"F{i}" for i in range(n)]
    diag_a = [f"A{i}" for i in range(2 * n - 1)]
    diag_b = [f"B{i}" for i in range(2 * n - 1)]
    items = items + diag_a + diag_b
    options = []
    for r in range(n):
        for c in range(n):
            options.append(((r, c), {f"R{r}", f"F{c}", f"A{r + c}", f"B{r - c + n - 1}"}))
    for d in diag_a + diag_b:
        options.append((("dummy", d), {d}))
    for sol in solve_exact_cover(items, options, find_all=True, limit=200):
        queens = [rid for rid in sol if rid[0] != "dummy"]
        if len(queens) == n:
            board = [0] * n
            for (r, c) in queens:
                board[r] = c
            return board
    return list(range(n))


def _svg(path, board, cell=48):
    n = len(board)
    ox, oy = 40, 70
    size = n * cell
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{ox*2+size}" height="{oy+size+30}" '
        f'viewBox="0 0 {ox*2+size} {oy+size+30}" font-family="monospace">',
        f'<rect width="{ox*2+size}" height="{oy+size+30}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">'
        f'A {n}-queens solution found by DLX exact cover</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'no two queens share a row, column, or diagonal</text>',
    ]
    for r in range(n):
        for c in range(n):
            x = ox + c * cell
            y = oy + r * cell
            light = (r + c) % 2 == 0
            fill = "#30363d" if light else "#161b22"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}"/>')
            if board[r] == c:
                parts.append(f'<circle cx="{x+cell/2:.0f}" cy="{y+cell/2:.0f}" r="{cell*0.32:.0f}" '
                             f'fill="#ffd43b"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
