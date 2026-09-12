"""Demo: the rope -- O(log n) string editing where a flat buffer would be O(n).

Builds a rope, performs middle inserts and deletes, shows the tree stays balanced, and compares the
work of a middle insert against a flat string. Draws the rope's tree structure.

    python examples/rope_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rope import Rope, _collect_leaves  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Rope: a balanced tree that edits huge strings in O(log n)\n")

    r = Rope("the quick brown fox")
    print(f"  start: {r.to_string()!r}  (length {len(r)}, tree height {r.height})")
    r = r.insert(10, "very ")
    print(f"  insert 'very ' at 10: {r.to_string()!r}")
    r = r.delete(0, 4)
    print(f"  delete [0,4):         {r.to_string()!r}")
    r = r + Rope(" jumps")
    print(f"  concat ' jumps':      {r.to_string()!r}\n")

    # balance under many edits
    big = Rope("x" * 20)
    for i in range(5000):
        big = big.insert(len(big) // 2, "ab")
    n = len(big)
    print(f"  after 5000 middle inserts: length {n}, tree height {big.height}")
    print(f"    ideal height ~log2(n) = {math.log2(n):.1f}; balanced = {big.is_balanced()}")
    print(f"    a flat string would have copied ~{n} chars on the LAST insert alone;")
    print(f"    the rope touched only ~{big.height} nodes on the path.\n")

    print("  Text lives only in the leaves; internal nodes store the left subtree's length so indexing,")
    print("  splitting, and joining walk a logarithmic path. Concatenation is O(1), edits are")
    print("  split-then-concat, and rebalancing keeps the tree from degenerating -- the structure a")
    print("  real text editor uses to stay responsive on huge files.")

    _svg(os.path.join(outdir, "rope.svg"), Rope("the_quick_brown_fox_jumps_over"))
    print(f"\n  wrote {os.path.join(outdir, 'rope.svg')}")


def _svg(path, rope, width=760, height=380):
    # layout the tree with a simple recursive positioner
    root = rope.root
    positions = {}
    leaf_x = [0]

    def layout(node, depth):
        if node is None:
            return
        if node.is_leaf():
            x = leaf_x[0]
            leaf_x[0] += 1
            positions[id(node)] = (x, depth, node)
            return x
        lx = layout(node.left, depth + 1)
        rx = layout(node.right, depth + 1)
        x = (lx + rx) / 2
        positions[id(node)] = (x, depth, node)
        return x

    layout(root, 0)
    if not positions:
        return
    maxx = max(p[0] for p in positions.values()) or 1
    maxd = max(p[1] for p in positions.values()) or 1
    pad = 40

    def sx(x):
        return pad + x / maxx * (width - 2 * pad)

    def sy(d):
        return 70 + d / maxd * (height - 130)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="16">'
        f'Rope tree: text in leaves (green), lengths in internal nodes (blue)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'{rope.to_string()!r}</text>',
    ]
    # edges
    def draw_edges(node):
        if node is None or node.is_leaf():
            return
        x, d, _ = positions[id(node)]
        for child in (node.left, node.right):
            if child is not None:
                cx, cd, _ = positions[id(child)]
                parts.append(f'<line x1="{sx(x):.1f}" y1="{sy(d):.1f}" x2="{sx(cx):.1f}" '
                             f'y2="{sy(cd):.1f}" stroke="#30363d" stroke-width="1"/>')
                draw_edges(child)
    draw_edges(root)
    # nodes
    for x, d, node in positions.values():
        cx, cy = sx(x), sy(d)
        if node.is_leaf():
            parts.append(f'<rect x="{cx-16:.1f}" y="{cy-11:.1f}" width="32" height="22" rx="3" '
                         f'fill="#06d6a0"/>')
            parts.append(f'<text x="{cx:.1f}" y="{cy+4:.1f}" fill="#0d1117" font-size="10" '
                         f'text-anchor="middle">{node.text}</text>')
        else:
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="13" fill="#161b22" '
                         f'stroke="#4dabf7" stroke-width="2"/>')
            parts.append(f'<text x="{cx:.1f}" y="{cy+4:.1f}" fill="#4dabf7" font-size="10" '
                         f'text-anchor="middle">{node.weight}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
