"""Demo: BK-tree fuzzy search -- spell-checking a typo without scanning the dictionary.

Builds a BK-tree over a word list, corrects typos, shows how much of the dictionary the
triangle-inequality pruning lets it skip, and draws the tree with its distance-labelled edges.

    python examples/bk_tree_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bk_tree import BKTree, brute_search, levenshtein  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("BK-tree: fuzzy string search that prunes with the triangle inequality\n")

    words = ["book", "books", "boo", "boon", "cook", "crook", "look", "loot", "boot",
             "cake", "cape", "care", "card", "cart", "cast", "case", "cave", "cove",
             "code", "core", "bore", "bone", "bane", "band", "bend", "bond"]
    bt = BKTree()
    bt.add_all(words)
    print(f"  dictionary of {len(bt)} words\n")

    print("  spell-check by nearest neighbour:")
    for typo in ("boook", "craok", "cae", "bora"):
        d, w = bt.nearest(typo)
        print(f"    {typo!r:8} -> {w!r} (edit distance {d})")
    print()

    print("  fuzzy search within a tolerance, and how much the pruning skips:")
    for q, tol in (("book", 1), ("care", 1), ("bone", 2)):
        got = bt.search(q, tol)
        matches = [w for _, w in got]
        print(f"    within {tol} of {q!r}: {matches}")
        print(f"      visited {bt.last_visited} of {len(bt)} nodes "
              f"({100*bt.last_visited/len(bt):.0f}%)")
    print()

    print("  Each node's children are indexed by their exact distance to it. A query at distance d only")
    print("  descends children whose edge label is in [d-tol, d+tol] -- every other subtree is provably")
    print("  too far and is skipped. Same answer as a full scan, a fraction of the work.")

    _svg(os.path.join(outdir, "bk_tree.svg"), bt)
    print(f"\n  wrote {os.path.join(outdir, 'bk_tree.svg')}")


def _svg(path, bt, width=760, height=440):
    # layout the tree; x by in-order-ish leaf spread, y by depth
    positions = {}
    leaf_x = [0]

    def layout(node, depth):
        if not node.children:
            x = leaf_x[0]
            leaf_x[0] += 1
            positions[id(node)] = (x, depth, node)
            return x
        xs = []
        for edge in sorted(node.children):
            xs.append(layout(node.children[edge], depth + 1))
        x = sum(xs) / len(xs)
        positions[id(node)] = (x, depth, node)
        return x

    layout(bt.root, 0)
    maxx = max(p[0] for p in positions.values()) or 1
    maxd = max(p[1] for p in positions.values()) or 1
    pad = 45

    def sx(x):
        return pad + x / maxx * (width - 2 * pad)

    def sy(d):
        return 70 + d / maxd * (height - 130)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="16">'
        f'BK-tree: edges labelled by edit distance between parent and child</text>',
    ]
    # edges with distance labels
    def draw_edges(node):
        px, py, _ = positions[id(node)]
        for edge in sorted(node.children):
            child = node.children[edge]
            cx, cy, _ = positions[id(child)]
            parts.append(f'<line x1="{sx(px):.1f}" y1="{sy(py):.1f}" x2="{sx(cx):.1f}" '
                         f'y2="{sy(cy):.1f}" stroke="#30363d" stroke-width="1"/>')
            mx, my = (sx(px) + sx(cx)) / 2, (sy(py) + sy(cy)) / 2
            parts.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="7" fill="#0d1117" '
                         f'stroke="#ff922b" stroke-width="1"/>')
            parts.append(f'<text x="{mx:.1f}" y="{my+3:.1f}" fill="#ff922b" font-size="9" '
                         f'text-anchor="middle">{edge}</text>')
            draw_edges(child)
    draw_edges(bt.root)
    # nodes
    for x, d, node in positions.values():
        cx, cy = sx(x), sy(d)
        col = "#ffd43b" if node is bt.root else "#4dabf7"
        parts.append(f'<rect x="{cx-24:.1f}" y="{cy-10:.1f}" width="48" height="20" rx="3" '
                     f'fill="#161b22" stroke="{col}" stroke-width="1.5"/>')
        parts.append(f'<text x="{cx:.1f}" y="{cy+4:.1f}" fill="#e6edf3" font-size="9" '
                     f'text-anchor="middle">{node.word}</text>')
    parts.append(f'<text x="20" y="{height-14}" fill="#8b949e" font-size="11">'
                 f'orange numbers are edit distances; a query descends only edges within its tolerance '
                 f'band</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
