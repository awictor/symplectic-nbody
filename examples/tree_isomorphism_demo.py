"""Demo: AHU tree isomorphism -- deciding when two trees are the same shape.

Shows the canonical string built bottom-up for a rooted tree, decides isomorphism between a tree and
its relabelled copy versus a genuinely different tree, and draws two trees side by side with their
centers marked and canonical forms compared.

    python examples/tree_isomorphism_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tree_isomorphism import (rooted_canonical, centers, unrooted_canonical,  # noqa: E402
                              isomorphic, brute_isomorphic)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("AHU tree isomorphism: a canonical string that is equal iff trees match\n")

    # tree A: a small rooted tree
    tA = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]
    print("  tree A edges:", tA)
    print(f"  rooted canonical form (root 0): {rooted_canonical(6, tA, 0)}")
    print("  -> each leaf is '()', each node sorts and wraps its children's forms\n")

    # tree B: A with vertices relabelled -> must be isomorphic
    relabel = {0: 5, 1: 2, 2: 4, 3: 0, 4: 1, 5: 3}
    tB = [(relabel[u], relabel[v]) for u, v in tA]
    print("  tree B = tree A with vertices shuffled:", tB)
    print(f"  isomorphic(A, B)?  {isomorphic(6, tA, 6, tB)}  (brute: {brute_isomorphic(6, tA, 6, tB)})")

    # tree C: same size, different shape (a path) -> not isomorphic
    tC = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    print(f"\n  tree C = a 6-vertex path:", tC)
    print(f"  isomorphic(A, C)?  {isomorphic(6, tA, 6, tC)}  (brute: {brute_isomorphic(6, tA, 6, tC)})")

    print(f"\n  unrooted canonical forms (rooted at the center for canonicity):")
    print(f"    A: {unrooted_canonical(6, tA)}")
    print(f"    B: {unrooted_canonical(6, tB)}   (equal to A -> isomorphic)")
    print(f"    C: {unrooted_canonical(6, tC)}   (differs -> not isomorphic)")

    print(f"\n  centers -- A: {centers(6, tA)}, C (path): {centers(6, tC)}")
    print("\n  Rooting at the center makes the form canonical: every tree has 1 or 2 centers (the")
    print("  middle of its longest path), found by peeling leaves. General GRAPH isomorphism has no")
    print("  known polynomial algorithm, but trees fall in linear time to this bottom-up hashing.")

    _svg(os.path.join(outdir, "tree_isomorphism.svg"), tA, tB, tC)
    print(f"\n  wrote {os.path.join(outdir, 'tree_isomorphism.svg')}")


def _layout(n, edges, root):
    """Simple layered layout: BFS depth sets y, order within a layer sets x."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    depth = {root: 0}
    order = [root]
    seen = {root}
    parent = {root: -1}
    i = 0
    while i < len(order):
        u = order[i]
        i += 1
        for v in sorted(adj[u]):
            if v not in seen:
                seen.add(v)
                depth[v] = depth[u] + 1
                parent[v] = u
                order.append(v)
    by_depth = {}
    for v in order:
        by_depth.setdefault(depth[v], []).append(v)
    pos = {}
    for d, verts in by_depth.items():
        k = len(verts)
        for j, v in enumerate(verts):
            pos[v] = ((j + 1) / (k + 1), d)
    return pos, parent


def _draw_tree(parts, n, edges, root, ox, oy, w, h, title, col):
    pos, parent = _layout(n, edges, root)
    maxd = max(d for _, d in pos.values()) or 1

    def sx(fx):
        return ox + fx * w

    def sy(d):
        return oy + (d / maxd) * h if maxd else oy

    parts.append(f'<text x="{ox + w/2:.0f}" y="{oy-14:.0f}" fill="{col}" font-size="13" '
                 f'text-anchor="middle">{title}</text>')
    ctr = set(centers(n, edges))
    for v in range(n):
        if parent.get(v, -1) != -1:
            x1, y1 = sx(pos[v][0]), sy(pos[v][1])
            x2, y2 = sx(pos[parent[v]][0]), sy(pos[parent[v]][1])
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                         f'stroke="#484f58" stroke-width="1.5"/>')
    for v in range(n):
        x, y = sx(pos[v][0]), sy(pos[v][1])
        fill = "#ffd43b" if v in ctr else col
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="{fill}"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+4:.1f}" fill="#0d1117" font-size="10" '
                     f'text-anchor="middle">{v}</text>')


def _svg(path, tA, tB, tC, width=780, height=420):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'A and B are the same tree relabelled (isomorphic); C is a different shape</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'yellow nodes are tree centers; AHU roots there and hashes bottom-up to a canonical form'
        f'</text>',
    ]
    _draw_tree(parts, 6, tA, 0, 40, 90, 200, 260, "A", "#4dabf7")
    _draw_tree(parts, 6, tB, 5, 290, 90, 200, 260, "B (= A relabelled)", "#06d6a0")
    _draw_tree(parts, 6, tC, 0, 560, 90, 190, 260, "C (path, different)", "#ff6b6b")
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
