"""Demo: Boruvka's MST built round by round, each component grabbing its cheapest outgoing edge.

Builds a Euclidean graph on random points, runs Boruvka round by round (instrumenting the merges),
reports how the component count halves each round, and draws the points with the MST edges coloured by
the round in which Boruvka added them.

    python examples/boruvka_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from boruvka import boruvka_mst, kruskal_weight  # noqa: E402
from union_find import UnionFind  # noqa: E402


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def _boruvka_rounds(n, edges):
    """Re-run Boruvka but record which round each edge was added in (for drawing)."""
    uf = UnionFind(n)
    total = 0.0
    rounds = []  # list of lists of (w,u,v) per round
    indexed = [(w, u, v, i) for i, (w, u, v) in enumerate(edges)]
    while uf.count() > 1:
        cheapest = {}
        for w, u, v, idx in indexed:
            ru, rv = uf.find(u), uf.find(v)
            if ru == rv:
                continue
            key = (w, idx)
            for r in (ru, rv):
                if r not in cheapest or key < cheapest[r][0]:
                    cheapest[r] = (key, (w, u, v))
        if not cheapest:
            break
        this_round = []
        for r, (key, (w, u, v)) in cheapest.items():
            if uf.union(u, v):
                total += w
                this_round.append((w, u, v))
        if not this_round:
            break
        rounds.append(this_round)
    return total, rounds


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Boruvka's algorithm: MST in synchronized rounds, each component grabbing its cheapest edge\n")

    rng = _lcg(2024)
    n = 30
    pts = [(rng() * 100, rng() * 100) for _ in range(n)]
    # complete Euclidean graph
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            edges.append((w, i, j))

    total, rounds = _boruvka_rounds(n, edges)
    kw = kruskal_weight(n, edges)

    print(f"  {n} points, complete Euclidean graph ({len(edges)} edges)\n")
    print(f"    {'round':>6}{'edges added':>14}{'components left':>18}")
    comp = n
    for r, edges_r in enumerate(rounds, 1):
        comp -= len(edges_r)
        print(f"    {r:>6}{len(edges_r):>14}{comp:>18}")
    print(f"\n  {len(rounds)} rounds for {n} vertices (Boruvka needs O(log V) ~ {math.ceil(math.log2(n))})")
    print(f"  MST weight: {total:.3f}   (Kruskal cross-check: {kw:.3f}, match: {abs(total-kw)<1e-9})")
    print(f"\n  Each round at least halves the component count -- that's the O(log V) round bound,")
    print(f"  and every round's cheapest-edge search is independent, so Boruvka parallelizes.")

    _svg(os.path.join(outdir, "boruvka.svg"), pts, rounds)
    print(f"\n  wrote {os.path.join(outdir, 'boruvka.svg')}")


def _svg(path, pts, rounds, width=760, height=440):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    pad = 34

    def sx(x):
        return pad + (width - 2 * pad) * (x - minx) / (maxx - minx + 1e-9)

    def sy(y):
        return height - pad - (height - 2 * pad - 20) * (y - miny) / (maxy - miny + 1e-9)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Boruvka MST: edges coloured by the round they were added (early = blue, late = red)</text>',
    ]
    colors = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#ff6b6b", "#b197fc"]
    for r, edges_r in enumerate(rounds):
        color = colors[min(r, len(colors) - 1)]
        for w, u, v in edges_r:
            parts.append(f'<line x1="{sx(pts[u][0]):.1f}" y1="{sy(pts[u][1]):.1f}" '
                         f'x2="{sx(pts[v][0]):.1f}" y2="{sy(pts[v][1]):.1f}" '
                         f'stroke="{color}" stroke-width="2"/>')
    for x, y in pts:
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="#e6edf3"/>')
    # legend
    ly = height - 40
    for r in range(len(rounds)):
        color = colors[min(r, len(colors) - 1)]
        parts.append(f'<rect x="{40 + r*70}" y="{ly}" width="12" height="4" fill="{color}"/>')
        parts.append(f'<text x="{56 + r*70}" y="{ly+5}" fill="#8b949e" font-size="9">rd {r+1}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
