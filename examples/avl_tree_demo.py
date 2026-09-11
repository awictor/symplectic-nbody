"""Demo: an AVL self-balancing tree keeping its height logarithmic.

Shows that inserting sorted keys -- the worst case that turns a naive BST into a linked list --
keeps an AVL tree's height at O(log n) via rotations, illustrates the four rotation cases, and
draws the balanced tree. Contrasts AVL height against the linear height a naive BST would reach.

    python examples/avl_tree_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from avl_tree import AVLTree  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("AVL tree: a binary search tree that rotates itself back into balance\n")

    print("  Inserting SORTED keys (the naive-BST worst case that degrades to a linked list):")
    print(f"    {'n':>6} {'AVL height':>11} {'naive BST':>11} {'log2(n)':>9}")
    for n in (15, 63, 255, 1023, 4095):
        t = AVLTree()
        for k in range(n):
            t.insert(k, k)
        print(f"    {n:>6} {t.height():>11} {n:>11} {math.log2(n):>9.1f}")
    print("    -> AVL height tracks log2(n); a naive BST would be n (a linear chain).")

    print("\n  The four rotation cases, each fixing one imbalance shape:")
    for seq, name, shape in [([3, 2, 1], "LL", "left-left  -> right rotation"),
                             ([1, 2, 3], "RR", "right-right-> left rotation"),
                             ([3, 1, 2], "LR", "left-right -> left then right"),
                             ([1, 3, 2], "RL", "right-left -> right then left")]:
        t = AVLTree()
        for k in seq:
            t.insert(k)
        print(f"    insert {seq} ({name}, {shape}): root rebalances to {t.root.key}, "
              f"balanced={t.is_balanced()}")

    # a worked example tree
    demo = AVLTree()
    for k in [50, 30, 70, 20, 40, 60, 80, 10, 25, 35]:
        demo.insert(k, k)
    print(f"\n  Example tree of {len(demo)} keys: height {demo.height()}, "
          f"balanced={demo.is_balanced()}, sorted keys {demo.keys()}")
    print(f"  range [25, 60]: {[k for k, _ in demo.range(25, 60)]}")

    print("\n  After every insert or delete, the tree checks the balance factor up the path and,")
    print("  where |left height - right height| > 1, rotates -- a constant-time pointer rewiring --")
    print("  to restore balance. Deterministic O(log n) worst case, where a skip list gets there")
    print("  probabilistically with coin flips instead of rotations.")

    _svg(os.path.join(outdir, "avl_tree.svg"), demo)
    print(f"\n  wrote {os.path.join(outdir, 'avl_tree.svg')}")


def _svg(path, tree, width=760, height=400):
    # assign x by in-order index, y by depth
    order = []

    def inorder(node, depth):
        if node is None:
            return
        inorder(node.left, depth + 1)
        order.append((node, depth))
        inorder(node.right, depth + 1)

    inorder(tree.root, 0)
    n = len(order)
    max_depth = max(d for _, d in order) if order else 0
    pos = {}
    x0, x1 = 45, width - 30
    ty, by = 80, height - 50
    for idx, (node, depth) in enumerate(order):
        x = x0 + (idx + 0.5) / n * (x1 - x0)
        y = ty + (depth / max(1, max_depth)) * (by - ty)
        pos[id(node)] = (x, y)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'AVL tree: height-balanced, every subtree within 1 level</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'in-order left-to-right, depth top-to-bottom; the tree stays log-deep no matter the '
        f'insert order</text>',
    ]

    def draw_edges(node):
        if node is None:
            return
        x, y = pos[id(node)]
        for child in (node.left, node.right):
            if child is not None:
                cx, cy = pos[id(child)]
                parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" '
                             f'stroke="#30363d" stroke-width="1.3"/>')
                draw_edges(child)

    draw_edges(tree.root)
    for node, depth in order:
        x, y = pos[id(node)]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" fill="#4dabf7" '
                     f'stroke="#0d1117" stroke-width="1.5"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+4:.1f}" fill="#0d1117" font-size="11" '
                     f'text-anchor="middle" font-weight="bold">{node.key}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
