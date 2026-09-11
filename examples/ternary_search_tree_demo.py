"""Demo: a ternary search tree for autocomplete and wildcard search.

Builds a TST from a small dictionary, shows prefix autocomplete, longest-prefix matching, and '.'
wildcard search, and draws the TST structure (left/right BST links vs the middle 'advance' links).

    python examples/ternary_search_tree_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ternary_search_tree import TernarySearchTree  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Ternary search tree: trie prefix power with BST space\n")

    words = ["cat", "cats", "car", "card", "care", "dog", "do", "dodge", "dot", "day", "ape", "app"]
    tst = TernarySearchTree()
    for i, w in enumerate(words):
        tst.insert(w, len(w))       # value = word length, just to show values

    print(f"  inserted {len(words)} words: {', '.join(words)}")
    print(f"  stored (sorted): {', '.join(tst.keys())}\n")

    print("  Autocomplete (keys_with_prefix):")
    for p in ["ca", "do", "app", "d"]:
        print(f"    '{p}' -> {tst.keys_with_prefix(p)}")

    print("\n  Longest prefix of a query (longest_prefix_of):")
    for q in ["cards", "doghouse", "dotted", "apex"]:
        print(f"    '{q}' -> '{tst.longest_prefix_of(q)}'")

    print("\n  Wildcard search ('.' matches any one letter):")
    for pat in ["c.r", "do.", "ca..", "..t"]:
        print(f"    '{pat}' -> {tst.wildcard(pat)}")

    print("\n  Each node holds one character and three children: left/right for the BST of")
    print("  alternatives at this position, and a middle link that advances to the next character")
    print("  (spelling a key, like a trie). One three-way comparison drives every operation.")

    _svg(os.path.join(outdir, "ternary_search_tree.svg"), tst)
    print(f"\n  wrote {os.path.join(outdir, 'ternary_search_tree.svg')}")


def _svg(path, tst, width=760, height=470):
    # lay out the TST with a simple recursive placement:
    #   x increases with depth-of-character-advance (middle links go right),
    #   left/right links spread vertically.
    positions = {}
    order = [0]  # a mutable y-cursor

    # assign y by an in-order-ish traversal of left/right at each middle level, x by char-depth
    def layout(node, depth):
        if node is None:
            return
        layout(node.left, depth)
        node_y = order[0]
        order[0] += 1
        positions[id(node)] = (depth, node_y, node)
        layout(node.mid, depth + 1)
        layout(node.right, depth)

    layout(tst.root, 0)
    if not positions:
        open(path, "w").write("<svg/>")
        return
    max_depth = max(p[0] for p in positions.values())
    max_y = max(p[1] for p in positions.values())
    x_step = (width - 120) / (max_depth + 1)
    y_step = (height - 110) / (max_y + 1)

    def px(depth):
        return 60 + depth * x_step

    def py(y):
        return 90 + y * y_step

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Ternary search tree</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'green = middle link (advance to next character), gray = left/right BST link; '
        f'gold ring = end of a key</text>',
    ]

    # edges first
    def draw_edges(node):
        if node is None:
            return
        d, y, _ = positions[id(node)]
        x0, y0 = px(d), py(y)
        for child, col, dd in ((node.left, "#8b949e", d), (node.right, "#8b949e", d),
                               (node.mid, "#06d6a0", d + 1)):
            if child is not None:
                dc, yc, _ = positions[id(child)]
                parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{px(dc):.1f}" y2="{py(yc):.1f}" '
                             f'stroke="{col}" stroke-width="1.6"/>')
        draw_edges(node.left)
        draw_edges(node.mid)
        draw_edges(node.right)

    draw_edges(tst.root)

    # nodes
    for d, y, node in positions.values():
        x, yy = px(d), py(y)
        if node.is_key:
            parts.append(f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="14" fill="#161b22" '
                         f'stroke="#ffd43b" stroke-width="2.5"/>')
        else:
            parts.append(f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="12" fill="#161b22" '
                         f'stroke="#4dabf7" stroke-width="1.8"/>')
        parts.append(f'<text x="{x:.1f}" y="{yy + 4:.1f}" fill="#e6edf3" font-size="13" '
                     f'text-anchor="middle" font-weight="bold">{node.ch}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
