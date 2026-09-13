"""Demo: red-black tree -- the balanced, order-statistic map behind std::map and TreeMap.

Builds a red-black tree, shows it stays balanced even on sorted input (which ruins a plain BST),
answers order-statistic queries (k-th smallest, rank), and draws the coloured tree.

    python examples/red_black_tree_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from red_black_tree import RedBlackTree, RED  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Red-black tree: guaranteed O(log n), with order statistics\n")

    # sorted insertion -- the adversarial case for a naive BST
    print("  inserting 1..N in sorted order (a plain BST would become a linked list):")
    for N in (100, 1000, 100000):
        t = RedBlackTree()
        for k in range(N):
            t.insert(k, k)
        print(f"    N={N:6d}: height {t.height():2d}  (bound 2*log2(N+1) = {2*math.log2(N+1):.1f}, "
              f"a BST would be {N}), balanced = {t.check_invariants()}")
    print()

    # order statistics
    t = RedBlackTree()
    data = [50, 20, 80, 10, 30, 70, 90, 5, 25, 60]
    for k in data:
        t.insert(k, k)
    print(f"  order-statistic queries on {sorted(data)}:")
    print(f"    3rd smallest (select 2): {t.select(2)}")
    print(f"    median (select {len(t)//2}): {t.select(len(t) // 2)}")
    print(f"    largest (select {len(t)-1}): {t.select(len(t) - 1)}")
    print(f"    rank of 55 (how many keys < 55): {t.rank(55)}")
    print(f"    rank of 90: {t.rank(90)}\n")

    print("  Four colour invariants -- root black, no red-red edge, equal black height on every")
    print("  root-to-null path -- force the longest path to be at most twice the shortest, so the")
    print("  tree self-balances with O(1) rotations per update. Augmenting each node with its subtree")
    print("  size turns it into an order-statistic tree: select and rank in O(log n).")

    # small tree for the picture
    t = RedBlackTree()
    for k in [10, 5, 15, 3, 7, 12, 18, 1, 4, 6, 8, 13, 20]:
        t.insert(k, k)
    _svg(os.path.join(outdir, "red_black_tree.svg"), t)
    print(f"\n  wrote {os.path.join(outdir, 'red_black_tree.svg')}")


def _svg(path, tree, width=760, height=380):
    positions = {}
    xcounter = [0]

    def layout(node, depth):
        if node is tree.nil:
            return
        layout(node.left, depth + 1)
        x = xcounter[0]
        xcounter[0] += 1
        positions[id(node)] = (x, depth, node)
        layout(node.right, depth + 1)

    layout(tree.root, 0)
    if not positions:
        return
    maxx = max(p[0] for p in positions.values()) or 1
    maxd = max(p[1] for p in positions.values()) or 1
    pad = 40

    def sx(x):
        return pad + x / maxx * (width - 2 * pad)

    def sy(d):
        return 70 + d / maxd * (height - 120)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="16">'
        f'Red-black tree: node colours enforce balance (labels show subtree size)</text>',
    ]
    # edges
    def draw_edges(node):
        if node is tree.nil:
            return
        px, py, _ = positions[id(node)]
        for child in (node.left, node.right):
            if child is not tree.nil:
                cx, cy, _ = positions[id(child)]
                parts.append(f'<line x1="{sx(px):.1f}" y1="{sy(py):.1f}" x2="{sx(cx):.1f}" '
                             f'y2="{sy(cy):.1f}" stroke="#30363d" stroke-width="1"/>')
                draw_edges(child)
    draw_edges(tree.root)
    # nodes
    for x, d, node in positions.values():
        cx, cy = sx(x), sy(d)
        fill = "#ff6b6b" if node.color == RED else "#30363d"
        stroke = "#ff6b6b" if node.color == RED else "#8b949e"
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="15" fill="{fill}" '
                     f'stroke="{stroke}" stroke-width="2"/>')
        parts.append(f'<text x="{cx:.1f}" y="{cy+4:.1f}" fill="#e6edf3" font-size="11" '
                     f'text-anchor="middle">{node.key}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{cy-19:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">sz{node.size}</text>')
    parts.append(f'<text x="20" y="{height-14}" fill="#8b949e" font-size="11">'
                 f'red and black nodes; every root-to-leaf path has the same number of black nodes</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
