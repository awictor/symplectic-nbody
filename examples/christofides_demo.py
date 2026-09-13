"""Demo: Christofides 1.5-approximation for metric TSP -- MST + matching + Euler shortcut.

Runs Christofides on a random Euclidean instance, compares its tour length to the exact Held-Karp
optimum and to nearest-neighbour, and draws the MST, the matching edges on the odd-degree vertices,
and the final tour.

    python examples/christofides_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from christofides import (  # noqa: E402
    christofides,
    euclidean_matrix,
    tour_length,
    _mst_edges,
    odd_degree_vertices,
    min_weight_perfect_matching,
)
from tsp import held_karp, nearest_neighbour  # noqa: E402


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Christofides: the 1.5-approximation for metric TSP\n")

    rng = _lcg(2024)
    n = 10
    pts = [(rng() * 100, rng() * 100) for _ in range(n)]
    D = euclidean_matrix(pts)

    tree = _mst_edges(n, D)
    odds = odd_degree_vertices(n, tree)
    match_edges, mcost = min_weight_perfect_matching(odds, D)
    ctour, clen = christofides(D)

    opt_tour, opt = held_karp(D)
    nn_tour = nearest_neighbour(D, 0)
    nn_len = tour_length(nn_tour, D)

    print(f"  {n} cities, Euclidean (metric) distances.\n")
    print(f"  MST edges: {len(tree)},  odd-degree vertices: {len(odds)}  {odds}")
    print(f"  min-weight matching on the odds: cost {mcost:.2f}\n")
    print(f"    {'method':<28}{'tour length':>12}{'vs optimum':>14}")
    print(f"    {'Held-Karp (exact optimum)':<28}{opt:>12.2f}{'1.000x':>14}")
    print(f"    {'Christofides (<=1.5x)':<28}{clen:>12.2f}{clen / opt:>13.3f}x")
    print(f"    {'nearest neighbour':<28}{nn_len:>12.2f}{nn_len / opt:>13.3f}x")
    print(f"\n  Christofides tour: {ctour}")
    print(f"  Guaranteed <= 1.5x optimum; here {clen / opt:.3f}x. The guarantee comes from")
    print("  MST (<= OPT) plus a min matching on the odd vertices (<= OPT/2).")

    _svg(os.path.join(outdir, "christofides.svg"), pts, tree, match_edges, ctour, odds)
    print(f"\n  wrote {os.path.join(outdir, 'christofides.svg')}")


def _svg(path, pts, tree, match_edges, tour, odds, width=760, height=420):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    pad = 40

    def sx(x):
        return pad + (width - 2 * pad) * (x - minx) / (maxx - minx + 1e-9)

    def sy(y):
        return height - pad - (height - 2 * pad - 20) * (y - miny) / (maxy - miny + 1e-9)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Christofides: MST (gray) + odd-vertex matching (orange) -&gt; shortcut tour (blue)</text>',
    ]
    # final tour (closed) drawn thick underneath-ish
    tpts = " ".join(f"{sx(pts[v][0]):.1f},{sy(pts[v][1]):.1f}" for v in tour)
    first = tour[0]
    parts.append(f'<polygon points="{tpts}" fill="none" stroke="#4dabf7" stroke-width="2" '
                 f'stroke-opacity="0.85"/>')
    # MST edges
    for u, v in tree:
        parts.append(f'<line x1="{sx(pts[u][0]):.1f}" y1="{sy(pts[u][1]):.1f}" '
                     f'x2="{sx(pts[v][0]):.1f}" y2="{sy(pts[v][1]):.1f}" '
                     f'stroke="#8b949e" stroke-width="1.5" stroke-dasharray="3 2"/>')
    # matching edges
    for u, v in match_edges:
        parts.append(f'<line x1="{sx(pts[u][0]):.1f}" y1="{sy(pts[u][1]):.1f}" '
                     f'x2="{sx(pts[v][0]):.1f}" y2="{sy(pts[v][1]):.1f}" '
                     f'stroke="#ff922b" stroke-width="2.5"/>')
    # vertices: odd ones highlighted
    oddset = set(odds)
    for i, (x, y) in enumerate(pts):
        color = "#ffd43b" if i in oddset else "#06d6a0"
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{sx(x)+7:.1f}" y="{sy(y)-6:.1f}" fill="#e6edf3" '
                     f'font-size="10">{i}</text>')
    # legend
    parts.append(f'<text x="{width-220}" y="{height-42}" fill="#ffd43b" font-size="10">'
                 f'yellow = odd-degree vertex</text>')
    parts.append(f'<text x="{width-220}" y="{height-28}" fill="#ff922b" font-size="10">'
                 f'orange = matching edge</text>')
    parts.append(f'<text x="{width-220}" y="{height-14}" fill="#4dabf7" font-size="10">'
                 f'blue = final tour</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
