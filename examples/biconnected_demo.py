"""Biconnected components demo: decompose a graph into blocks colored by robustness (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import biconnected as B


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
RED = "#ff6b6b"
PALETTE = ["#4dabf7", "#06d6a0", "#b197fc", "#ff922b", "#ffd43b", "#63e6be", "#faa2c1"]


def main(outdir=None):
    # a graph with two triangles bridged through a chain -- several blocks + cut vertices
    edges = [(0, 1), (1, 2), (2, 0),          # triangle A
             (2, 3),                           # bridge edge
             (3, 4), (4, 5), (5, 3),           # triangle B
             (5, 6), (6, 7),                    # a path tail
             (1, 8), (8, 9), (9, 1)]            # triangle C off vertex 1
    n = 10
    blocks, art = B.biconnected_components(n, edges)

    lines = []
    lines.append("Biconnected components: a graph's blocks and cut vertices")
    lines.append("=" * 58)
    lines.append(f"{n} vertices, {len(edges)} edges")
    lines.append(f"biconnected blocks: {len(blocks)}")
    lines.append(f"articulation points (cut vertices): {sorted(art)}")
    lines.append("")
    lines.append("blocks (each survives any single-vertex failure internally):")
    for i, blk in enumerate(blocks):
        verts = sorted(set(v for e in blk for v in e))
        kind = "cycle/2-connected" if len(blk) >= 3 else "bridge edge"
        lines.append(f"  block {i}: vertices {verts}  ({len(blk)} edges, {kind})")
    lines.append("")
    # verify
    total = sum(len(b) for b in blocks)
    lines.append(f"edges partitioned: {total} == {len(edges)}? {total == len(edges)}")
    lines.append(f"articulation matches brute-force deletion test: "
                 f"{art == B._brute_articulation(n, edges)}")
    lines.append("")
    lines.append("A cut vertex belongs to several blocks -- it is the fragile joint where the graph")
    lines.append("would split. Everything inside a block is robust to any single node loss.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # layout vertices on a circle; draw edges colored by block; cut vertices ringed red
        W, H = 600, 560
        cx, cy, r = 300, 300, 210
        pos = {}
        for v in range(n):
            ang = 2 * math.pi * v / n - math.pi / 2
            pos[v] = (cx + r * math.cos(ang), cy + r * math.sin(ang))

        # map each edge to its block index
        edge_block = {}
        for bi, blk in enumerate(blocks):
            for e in blk:
                edge_block[frozenset(e)] = bi

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="30" y="30" fill="{TEXT}" font-size="15">'
                 f'Biconnected blocks (colored) and cut vertices (red ring)</text>')
        for (u, v) in edges:
            bi = edge_block.get(frozenset((u, v)), 0)
            col = PALETTE[bi % len(PALETTE)]
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            s.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="{col}" stroke-width="3"/>')
        for v in range(n):
            x, y = pos[v]
            ring = RED if v in art else BG
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" fill="#161b22" '
                     f'stroke="{ring}" stroke-width="{3 if v in art else 1.5}"/>')
            s.append(f'<text x="{x:.1f}" y="{y+4:.1f}" fill="{TEXT}" font-size="11" '
                     f'text-anchor="middle">{v}</text>')
        # legend
        ly = H - 60
        s.append(f'<circle cx="40" cy="{ly}" r="10" fill="#161b22" stroke="{RED}" stroke-width="3"/>')
        s.append(f'<text x="58" y="{ly+4}" fill="{TEXT}" font-size="11">cut vertex (articulation point)</text>')
        s.append(f'<text x="30" y="{ly+26}" fill="{GRAY}" font-size="10">'
                 f'edge color = biconnected block; {len(blocks)} blocks meet at {len(art)} cut vertices</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "biconnected.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
