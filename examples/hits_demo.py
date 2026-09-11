"""Demo: HITS hubs and authorities on a small web.

Ranks a directed graph by both HITS roles -- authorities (definitive sources many good hubs point
to) and hubs (good link lists that point to many authorities) -- and contrasts them with PageRank's
single score, showing that a good hub and a good authority are genuinely different pages. Draws the
graph twice, node size = hub score then = authority score.

    python examples/hits_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hits import hits, top_hubs, top_authorities  # noqa: E402
from pagerank import pagerank  # noqa: E402


# a small web: some pages are curated link lists (hubs), others are cited sources (authorities)
GRAPH = {
    "portal":  ["news", "wiki", "journal", "blog"],   # a link directory -> strong hub
    "reader":  ["news", "wiki", "journal"],
    "student": ["wiki", "journal"],
    "fan":     ["blog", "news"],
    "news":    ["wiki"],                                # sources occasionally cite each other
    "journal": ["wiki"],
    "blog":    ["news"],
}


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    hub, auth = hits(GRAPH)
    pr = pagerank(GRAPH)

    print("HITS: hubs and authorities (two scores per node)\n")
    print(f"  {len(hub)} pages, {sum(len(v) for v in GRAPH.values())} links\n")

    print("  Top HUBS (good lists of links -- point to many authorities):")
    for node, sc in top_hubs(hub, 4):
        print(f"    {node:>8}: {sc:.3f}")
    print("\n  Top AUTHORITIES (definitive sources -- linked by many good hubs):")
    for node, sc in top_authorities(auth, 4):
        print(f"    {node:>8}: {sc:.3f}")

    print("\n  Hubs and authorities are DIFFERENT roles -- compare the two rankings and PageRank:")
    print(f"    {'page':>8} {'hub':>7} {'authority':>10} {'pagerank':>9}")
    for node in sorted(hub, key=lambda n: -auth[n]):
        print(f"    {node:>8} {hub[node]:>7.3f} {auth[node]:>10.3f} {pr[node]:>9.3f}")
    top_hub = top_hubs(hub, 1)[0][0]
    top_auth = top_authorities(auth, 1)[0][0]
    print(f"\n  Best hub is '{top_hub}' (a link directory); best authority is '{top_auth}' "
          f"(what everyone cites).")
    print("  PageRank collapses this into one score; HITS keeps the two roles distinct.")

    print("\n  A good authority is pointed to by good hubs; a good hub points to good authorities.")
    print("  That mutual recursion is resolved by power iteration -- authorities are the dominant")
    print("  eigenvector of A'A, hubs of AA'. Kleinberg's HITS ran per query; PageRank runs once")
    print("  globally. Both turn the link graph into a ranking by linear algebra.")

    _svg(os.path.join(outdir, "hits.svg"), GRAPH, hub, auth)
    print(f"\n  wrote {os.path.join(outdir, 'hits.svg')}")


def _svg(path, graph, hub, auth, width=760, height=430):
    nodes = sorted(hub, key=str)
    n = len(nodes)
    # circular layout
    import math as m
    cx_l, cy_l, r_l = 195, 250, 135

    def positions(cx):
        pos = {}
        for i, nd in enumerate(nodes):
            ang = 2 * m.pi * i / n - m.pi / 2
            pos[nd] = (cx + r_l * m.cos(ang), cy_l + r_l * m.sin(ang))
        return pos

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'HITS: node size = hub score (left) vs authority score (right)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'the biggest hub (a link list) and the biggest authority (a cited source) are '
        f'different nodes</text>',
        '<defs><marker id="ah" markerWidth="7" markerHeight="7" refX="6" refY="2.5" '
        'orient="auto"><path d="M0,0 L6,2.5 L0,5 Z" fill="#484f58"/></marker></defs>',
    ]

    def panel(cx, scores, color, title):
        pos = positions(cx)
        top = max(scores.values()) or 1.0
        # edges
        for src, outs in graph.items():
            x0, y0 = pos[src]
            for dst in outs:
                x1, y1 = pos[dst]
                dx, dy = x1 - x0, y1 - y0
                L = m.hypot(dx, dy) or 1.0
                rs = 5 + scores[src] / top * 20
                rd = 5 + scores[dst] / top * 20
                ax, ay = x0 + dx / L * rs, y0 + dy / L * rs
                bx, by = x1 - dx / L * (rd + 5), y1 - dy / L * (rd + 5)
                parts.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                             f'stroke="#30363d" stroke-width="1" marker-end="url(#ah)"/>')
        for nd in nodes:
            x, y = pos[nd]
            rad = 5 + scores[nd] / top * 20
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}" fill="{color}" '
                         f'stroke="#0d1117" stroke-width="1.4" opacity="0.9"/>')
            parts.append(f'<text x="{x:.1f}" y="{y+rad+11:.1f}" fill="#e6edf3" font-size="9" '
                         f'text-anchor="middle">{nd}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{height-16:.1f}" fill="{color}" font-size="12" '
                     f'text-anchor="middle">{title}</text>')

    panel(195, hub, "#4dabf7", "hubs (link lists)")
    panel(width - 195, auth, "#ff6b6b", "authorities (cited sources)")
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
