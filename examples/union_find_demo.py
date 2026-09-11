"""Demo: Union-Find -- connectivity in near-constant time, and Kruskal's MST.

Feeds connection facts into a disjoint-set structure and watches the component count drop, then
runs Kruskal's minimum-spanning-tree on a small weighted graph. Draws the merging of components
and the MST edges chosen (cheapest-first, skipping any that would close a cycle).

    python examples/union_find_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from union_find import UnionFind, kruskal_mst  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Union-Find: merge groups and query connectivity in ~O(1) amortized\n")
    uf = UnionFind(10)
    facts = [(0, 1), (2, 3), (4, 5), (1, 2), (6, 7), (5, 8), (3, 6)]
    print(f"  start: {uf.count()} singletons")
    for a, b in facts:
        merged = uf.union(a, b)
        tag = "merge" if merged else "already"
        print(f"    connect {a}-{b}: {tag:>7} -> {uf.count()} components")
    print(f"\n  final components: {uf.components()}")
    print(f"  connected(0, 8)? {uf.connected(0, 8)}   connected(0, 9)? {uf.connected(0, 9)}")
    print(f"  size of 0's group: {uf.set_size(0)}\n")

    # Kruskal MST on a weighted graph
    edges = [(7, 0, 1), (5, 0, 3), (8, 1, 2), (9, 1, 3), (7, 1, 4),
             (5, 2, 4), (15, 3, 4), (6, 3, 5), (8, 4, 5), (9, 4, 6), (11, 5, 6)]
    total, chosen = kruskal_mst(7, edges)
    print("  Kruskal's MST (add cheapest edge that joins two components):")
    print(f"  {'weight':>7}{'edge':>10}")
    for w, u, v in chosen:
        print(f"  {w:>7}   {u}--{v}")
    print(f"  total spanning-tree weight = {total}  ({len(chosen)} edges for 7 nodes)")
    print("\n  A cycle is exactly two endpoints already in the same set, so Union-Find spots it")
    print("  in constant time -- which is the whole of Kruskal. It also drives image")
    print("  segmentation, percolation, and 'friend circle' / account-merge problems.")

    _svg(os.path.join(outdir, "union_find.svg"), edges, chosen)
    print(f"\n  wrote {os.path.join(outdir, 'union_find.svg')}")


def _svg(path, edges, chosen, w=760, h=390):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Union-Find &amp; Kruskal: build a minimum spanning tree</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'graph with all edges, MST edges highlighted (left); '
        f'components merging as edges are added (right)</text>',
    ]

    # node layout: 7 nodes on a circle
    n = 7
    cx, cy, r = 195, 210, 120
    pos = []
    for i in range(n):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        pos.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    chosen_set = {(u, v) for _, u, v in chosen} | {(v, u) for _, u, v in chosen}

    # draw all edges (thin gray), then MST edges (thick green)
    for wt, u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        is_mst = (u, v) in chosen_set
        col = "#06d6a0" if is_mst else "#30363d"
        sw = 3 if is_mst else 1
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="{col}" stroke-width="{sw}"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        parts.append(f'<text x="{mx:.1f}" y="{my:.1f}" fill="{"#06d6a0" if is_mst else "#6e7681"}" '
                     f'font-size="9" text-anchor="middle">{wt}</text>')
    for i, (x, y) in enumerate(pos):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" fill="#161b22" stroke="#4dabf7" '
                     f'stroke-width="1.5"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+4:.1f}" fill="#4dabf7" font-size="11" '
                     f'text-anchor="middle">{i}</text>')
    parts.append(f'<text x="{cx:.1f}" y="{cy+r+50:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">green = MST edges (cheapest, cycle-free)</text>')

    # right: component count as edges are added in Kruskal order
    rx0, rx1 = 430, w - 30
    ry0, ry1 = h - 60, 80
    order = sorted(edges)
    uf = UnionFind(n)
    counts = [n]
    for wt, u, v in order:
        uf.union(u, v)
        counts.append(uf.count())
    steps = len(counts)

    def RX(i):
        return rx0 + i / (steps - 1) * (rx1 - rx0)

    def RY(c):
        return ry0 - (c - 1) / (n - 1) * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    pts = " ".join(f"{RX(i):.1f},{RY(c):.1f}" for i, c in enumerate(counts))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#ff922b" stroke-width="2.5"/>')
    for i, c in enumerate(counts):
        parts.append(f'<circle cx="{RX(i):.1f}" cy="{RY(c):.1f}" r="2.5" fill="#ff922b"/>')
    parts.append(f'<line x1="{rx0}" y1="{RY(1):.1f}" x2="{rx1}" y2="{RY(1):.1f}" '
                 f'stroke="#21262d" stroke-width="1"/>')
    parts.append(f'<text x="{rx1-2:.1f}" y="{RY(1)-4:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">1 (connected)</text>')
    for c in (1, n):
        parts.append(f'<text x="{rx0-6:.1f}" y="{RY(c)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{c}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">edges added -> component count</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
