"""Demo: the traveling salesman problem -- exact Held-Karp and the 2-opt heuristic.

Solves a small instance exactly with Held-Karp, then shows 2-opt untangling a nearest-neighbour tour
on a larger instance where exact solving is infeasible. Draws the nearest-neighbour tour (with
crossings) beside the 2-opt tour (crossings removed).

    python examples/tsp_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tsp import (euclidean_matrix, held_karp, nearest_neighbour, two_opt, tour_length)  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Traveling salesman: exact Held-Karp DP and the 2-opt heuristic\n")

    state = 20260911

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    # small instance: exact solve
    small = [(rng() * 100, rng() * 100) for _ in range(11)]
    Ds = euclidean_matrix(small)
    hk_tour, hk_len = held_karp(Ds)
    nn_s = nearest_neighbour(Ds)
    print(f"  11 cities (Held-Karp exact, examining subsets not the 3.6M tours):")
    print(f"    nearest-neighbour tour: {tour_length(nn_s, Ds):.1f}")
    print(f"    Held-Karp optimum:      {hk_len:.1f}")
    print(f"    2-opt from NN:          {tour_length(two_opt(nn_s, Ds), Ds):.1f}")

    # larger instance: 2-opt only
    n = 60
    pts = [(rng() * 100, rng() * 100) for _ in range(n)]
    D = euclidean_matrix(pts)
    nn = nearest_neighbour(D)
    nn_len = tour_length(nn, D)
    opt2 = two_opt(nn, D)
    opt2_len = tour_length(opt2, D)
    print(f"\n  {n} cities ((n-1)!/2 = astronomically many tours; 2-opt local search):")
    print(f"    nearest-neighbour: {nn_len:.1f}")
    print(f"    after 2-opt:       {opt2_len:.1f}  ({100*(1-opt2_len/nn_len):.1f}% shorter)")
    print(f"    2-opt removes the self-crossings a good tour never has")

    print("\n  Held-Karp builds shortest paths over every (subset, last-city) pair, reusing")
    print("  subproblems for O(n^2 2^n) instead of O(n!). 2-opt repeatedly reverses a segment")
    print("  between two edges whenever that shortens the tour -- fast, and within a few percent.")

    _svg(os.path.join(outdir, "tsp.svg"), pts, nn, opt2, nn_len, opt2_len)
    print(f"\n  wrote {os.path.join(outdir, 'tsp.svg')}")


def _svg(path, pts, nn_tour, opt_tour, nn_len, opt_len, width=760, height=430):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'TSP: nearest-neighbour (left, crossings) vs 2-opt (right, untangled)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'2-opt reverses segments to remove crossings; the tour shrinks from {nn_len:.0f} to {opt_len:.0f}</text>',
    ]

    def panel(tour, ox, color, label):
        # scale points into a 340-wide panel
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        sx = 300 / (maxx - minx + 1e-9)
        sy = 300 / (maxy - miny + 1e-9)
        s = min(sx, sy)

        def tx(x):
            return ox + 20 + (x - minx) * s

        def ty(y):
            return 80 + (y - miny) * s

        pts_str = " ".join(f"{tx(pts[c][0]):.1f},{ty(pts[c][1]):.1f}" for c in tour)
        first = tour[0]
        segs = pts_str + f" {tx(pts[first][0]):.1f},{ty(pts[first][1]):.1f}"
        out = [f'<polyline points="{segs}" fill="none" stroke="{color}" stroke-width="1.4"/>']
        for c in tour:
            out.append(f'<circle cx="{tx(pts[c][0]):.1f}" cy="{ty(pts[c][1]):.1f}" r="2.5" fill="#e6edf3"/>')
        out.append(f'<text x="{ox+170:.0f}" y="{height-20}" fill="{color}" font-size="12" '
                   f'text-anchor="middle">{label}</text>')
        return out

    parts += panel(nn_tour, 0, "#ff6b6b", f"nearest-neighbour ({nn_len:.0f})")
    parts += panel(opt_tour, 380, "#06d6a0", f"2-opt ({opt_len:.0f})")
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
