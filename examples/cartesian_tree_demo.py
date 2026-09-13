"""Demo: the Cartesian tree turning an array into a tree where range-minima are ancestors.

Builds the Cartesian tree of a small sequence, prints its structure, and demonstrates the central
equivalence: the minimum of a[i..j] is the lowest common ancestor of nodes i and j. Draws the tree
with nodes placed at their array position (x) and value (depth), so the min-heap and BST-on-position
structure are both visible.

    python examples/cartesian_tree_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cartesian_tree import build_cartesian_tree, CartesianRMQ, brute_min_index  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Cartesian tree: range-minimum-query becomes lowest-common-ancestor\n")

    a = [9, 3, 7, 1, 8, 5, 6, 2, 4]
    root, left, right, parent = build_cartesian_tree(a)
    n = len(a)

    print(f"  sequence: {a}")
    print(f"  Cartesian tree (min-heap on value, BST on position):")
    print(f"    root = index {root} (value {a[root]}, the global minimum)")
    for i in range(n):
        lc = f"L={left[i]}" if left[i] != -1 else "L=."
        rc = f"R={right[i]}" if right[i] != -1 else "R=."
        print(f"    node {i} (val {a[i]}): {lc} {rc}")

    crmq = CartesianRMQ(a)
    print(f"\n  Range-minimum queries via LCA on the tree:")
    print(f"    {'range':>10}{'min idx':>9}{'min val':>9}{'brute':>7}")
    for (l, r) in [(0, 3), (2, 6), (4, 8), (1, 7), (5, 8), (0, 8)]:
        mi = crmq.min_index(l, r)
        print(f"    a[{l}..{r}]  {mi:>7}{a[mi]:>9}{brute_min_index(a, l, r):>7}")

    print(f"\n  Every query is the LCA of its two endpoints -- so O(n) preprocessing plus an O(1)")
    print(f"  LCA structure gives O(1) range-minimum, the classic RMQ<->LCA equivalence.")

    _svg(os.path.join(outdir, "cartesian_tree.svg"), a, root, left, right, parent)
    print(f"\n  wrote {os.path.join(outdir, 'cartesian_tree.svg')}")


def _svg(path, a, root, left, right, parent, width=760, height=420):
    n = len(a)
    vmax = max(a)
    vmin = min(a)
    ox, oy, ow, oh = 50, 55, width - 90, height - 110

    def px(i):
        return ox + ow * i / (n - 1) if n > 1 else ox + ow / 2

    def py(v):
        # smaller value = higher up (closer to root)
        return oy + oh * (v - vmin) / (vmax - vmin) if vmax > vmin else oy

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Cartesian tree: x = array position (BST), y = value (min-heap, root on top)</text>',
    ]
    # edges
    for i in range(n):
        for ch in (left[i], right[i]):
            if ch != -1:
                parts.append(f'<line x1="{px(i):.1f}" y1="{py(a[i]):.1f}" '
                             f'x2="{px(ch):.1f}" y2="{py(a[ch]):.1f}" '
                             f'stroke="#484f58" stroke-width="1.5"/>')
    # nodes
    for i in range(n):
        color = "#06d6a0" if i == root else "#4dabf7"
        parts.append(f'<circle cx="{px(i):.1f}" cy="{py(a[i]):.1f}" r="13" fill="{color}"/>')
        parts.append(f'<text x="{px(i):.1f}" y="{py(a[i])+4:.1f}" fill="#0d1117" font-size="11" '
                     f'text-anchor="middle">{a[i]}</text>')
        parts.append(f'<text x="{px(i):.1f}" y="{oy+oh+16:.0f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">i={i}</text>')
    parts.append(f'<text x="{px(root):.1f}" y="{py(a[root])-18:.1f}" fill="#06d6a0" font-size="10" '
                 f'text-anchor="middle">root (global min)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
