"""Demo: Gomory-Hu tree -- all-pairs min cuts of a graph packed into a single tree.

Builds the tree for a small weighted network, prints the full all-pairs min-cut table read off it,
and draws the original graph beside its Gomory-Hu tree.

    python examples/gomory_hu_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gomory_hu import gomory_hu_tree, all_pairs_min_cuts, min_cut_query  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    # A "dumbbell": two well-connected clusters joined by a thin waist.
    # Cluster A = {0,1,2}, cluster B = {3,4,5}, joined by edges (2,3) and (2,4).
    labels = ["A0", "A1", "A2", "B3", "B4", "B5"]
    edges = [
        (0, 1, 6), (1, 2, 6), (0, 2, 6),   # cluster A, heavy
        (3, 4, 6), (4, 5, 6), (3, 5, 6),   # cluster B, heavy
        (2, 3, 2), (2, 4, 1),              # thin waist: total 3 across
    ]
    n = 6

    print("Gomory-Hu tree: every pair's min cut from n-1 max-flow calls\n")
    print("  Graph: two heavy triangles A={A0,A1,A2}, B={B3,B4,B5}")
    print("  joined by a thin waist (2->3 weight 2, 2->4 weight 1, total 3 across).\n")

    tree = gomory_hu_tree(n, edges)
    print("  Gomory-Hu tree edges (u, parent, cut-weight):")
    for u, p, w in tree:
        print(f"    {labels[u]:>3} -- {labels[p]:<3}  weight {w}")

    table = all_pairs_min_cuts(n, tree)
    print("\n  All-pairs minimum-cut table (read as tree path-minimum):\n")
    header = "       " + " ".join(f"{labels[j]:>4}" for j in range(n))
    print(header)
    for i in range(n):
        row = "  ".join(f"{table[i][j]:>4}" if i != j else "   ." for j in range(n))
        print(f"    {labels[i]:>3} {row}")

    print("\n  Note: any within-cluster pair costs 12 (must sever two heavy edges), while any")
    print("  A-to-B pair costs 3 -- the waist. The tree recovers all 15 cuts from 5 flow calls.")

    _svg(os.path.join(outdir, "gomory_hu.svg"), n, labels, edges, tree)
    print(f"\n  wrote {os.path.join(outdir, 'gomory_hu.svg')}")


def _svg(path, n, labels, edges, tree, width=760, height=430):
    # Fixed layout: two triangles left and right.
    pos = {
        0: (150, 110), 1: (90, 230), 2: (210, 230),
        3: (550, 230), 4: (670, 230), 5: (610, 110),
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Original graph (left) and its Gomory-Hu tree (right)</text>',
    ]

    def node(cx, cy, label, col):
        return (
            f'<circle cx="{cx}" cy="{cy}" r="16" fill="#161b22" stroke="{col}" stroke-width="2"/>'
            f'<text x="{cx}" y="{cy+4}" fill="{col}" font-size="11" text-anchor="middle">{label}</text>'
        )

    # ---- left panel: original graph -----------------------------------------------------
    for u, v, w in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        thin = w <= 2
        col = "#ff6b6b" if thin else "#30363d"
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
                     f'stroke-width="{1.5 if thin else 3}"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        parts.append(f'<text x="{mx:.0f}" y="{my-4:.0f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{w}</text>')
    for i in range(n):
        cx, cy = pos[i]
        parts.append(node(cx, cy, labels[i], "#4dabf7"))
    parts.append('<text x="150" y="330" fill="#8b949e" font-size="10" text-anchor="middle">'
                 'red = thin waist edges</text>')

    # ---- right panel: the tree, laid out as a simple horizontal chain -------------------
    # Order vertices by tree DFS from 0 for a readable line.
    adj = {i: [] for i in range(n)}
    for u, p, w in tree:
        adj[u].append((p, w))
        adj[p].append((u, w))
    order = []
    seen = set()
    stack = [0]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        order.append(u)
        for v, _ in adj[u]:
            if v not in seen:
                stack.append(v)
    tx0, ty = 430, 200
    step = (width - tx0 - 40) / max(1, n - 1)
    tpos = {u: (tx0 + k * step, ty + (30 if k % 2 else -30)) for k, u in enumerate(order)}
    for u, p, w in tree:
        x1, y1 = tpos[u]
        x2, y2 = tpos[p]
        parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                     f'stroke="#06d6a0" stroke-width="2"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        parts.append(f'<text x="{mx:.0f}" y="{my-4:.0f}" fill="#ffd43b" font-size="10" '
                     f'text-anchor="middle">{w}</text>')
    for i in range(n):
        cx, cy = tpos[i]
        parts.append(node(cx, cy, labels[i], "#06d6a0"))
    parts.append('<text x="590" y="330" fill="#8b949e" font-size="10" text-anchor="middle">'
                 'path-min = min cut of any pair</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
