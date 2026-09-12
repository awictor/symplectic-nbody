"""Demo: Bron-Kerbosch -- finding every fully-connected group (maximal clique) in a network.

Enumerates the maximal cliques of a small social network, reports the largest, and draws the graph
with the maximum clique highlighted.

    python examples/bron_kerbosch_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bron_kerbosch import (maximal_cliques, maximum_clique,  # noqa: E402
                           brute_maximal_cliques)

NAMES = ["Ana", "Bo", "Cy", "Di", "Ed", "Fi", "Gu"]
# friendships (undirected)
EDGES = [
    (0, 1), (0, 2), (1, 2),           # Ana-Bo-Cy triangle
    (1, 3), (2, 3), (1, 2),           # + Di connected to Bo, Cy -> {Bo,Cy,Di} and bigger
    (3, 4), (4, 5), (3, 5),           # Di-Ed-Fi triangle
    (5, 6),                            # Fi-Gu edge
]
N = 7


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    cliques = maximal_cliques(N, EDGES)
    best, size = maximum_clique(N, EDGES)

    print("Bron-Kerbosch: every maximal clique (fully-connected group) in a friend network\n")
    print(f"  {N} people, {len(set((min(u,v),max(u,v)) for u,v in EDGES))} friendships\n")

    print(f"  {len(cliques)} maximal cliques (groups where everyone knows everyone, not extendable):")
    for c in sorted(cliques, key=lambda s: (-len(s), sorted(s))):
        names = ", ".join(NAMES[v] for v in sorted(c))
        print(f"    {{{names}}}")

    print(f"\n  largest group (maximum clique): {{{', '.join(NAMES[v] for v in sorted(best))}}} "
          f"-- {size} people")

    brute = set(brute_maximal_cliques(N, EDGES))
    print(f"\n  matches brute-force subset check: {set(cliques) == brute}")
    print("\n  Bron-Kerbosch recurses over R (clique so far), P (extenders), X (already-tried); a PIVOT")
    print("  in P union X lets it skip candidates that are the pivot's neighbours, since every maximal")
    print("  clique already contains the pivot or one of its non-neighbours -- pruning huge swaths of")
    print("  the search. The number of maximal cliques can reach 3^(n/3), so this pruning is essential.")

    _svg(os.path.join(outdir, "bron_kerbosch.svg"), best)
    print(f"\n  wrote {os.path.join(outdir, 'bron_kerbosch.svg')}")


def _svg(path, best, width=720, height=420):
    # circular layout
    cx, cy, r = width / 2, height / 2 + 10, 150
    pos = {}
    for v in range(N):
        ang = 2 * math.pi * v / N - math.pi / 2
        pos[v] = (cx + r * math.cos(ang), cy + r * math.sin(ang))
    best_set = set(best)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Maximal cliques: the largest fully-connected group highlighted in green</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'green edges/nodes are the maximum clique; grey edges are the rest of the friendships</text>',
    ]

    for u, v in EDGES:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        inside = u in best_set and v in best_set
        col = "#06d6a0" if inside else "#484f58"
        wid = 3.5 if inside else 1.3
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="{col}" stroke-width="{wid}"/>')

    for v, (x, y) in pos.items():
        fill = "#06d6a0" if v in best_set else "#4dabf7"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="22" fill="{fill}"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+5:.1f}" fill="#0d1117" font-size="12" '
                     f'text-anchor="middle" font-weight="bold">{NAMES[v]}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
