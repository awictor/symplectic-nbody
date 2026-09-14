"""Suffix tree demo: build the tree for 'banana', draw it, and answer substring queries (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from suffix_tree import SuffixTree
import suffix_array as SA


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
YELLOW = "#ffd43b"
EDGE = "#484f58"


def _layout(tree):
    """Assign (depth-in-chars, y-slot) to each node for a left-to-right tree drawing."""
    text = tree.text

    def edge_label(node):
        end = tree._edge_end(node)
        return text[node.start:end].replace(text[-1], "$")

    # collect leaves in DFS order for y placement
    positions = {}
    order = [0]

    def assign_y(node):
        if not node.children:
            positions[id(node)] = order[0]
            order[0] += 1
            return positions[id(node)]
        ys = []
        for c in sorted(node.children):
            ys.append(assign_y(node.children[c]))
        y = sum(ys) / len(ys)
        positions[id(node)] = y
        return y

    assign_y(tree.root)
    return positions, edge_label


def _svg(path, tree, word):
    positions, edge_label = _layout(tree)
    text = tree.text
    W, H = 720, 400
    ml, mt = 40, 50
    xstep = 74
    ystep = 34
    nleaves = sum(1 for _ in _iter_leaves(tree.root))

    def px(depth):
        return ml + depth * xstep

    def py(slot):
        return mt + slot * ystep

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="monospace">']
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    s.append(f'<text x="{ml}" y="26" fill="{TEXT}" font-size="15">'
             f'Suffix tree of "{word}$" (edges labelled by substring)</text>')

    def draw(node, depth):
        x0 = px(depth)
        y0 = py(positions[id(node)])
        for c in sorted(node.children):
            child = node.children[c]
            lbl = edge_label(child)
            cdepth = depth + len(lbl)
            x1 = px(cdepth)
            y1 = py(positions[id(child)])
            s.append(f'<line x1="{x0:.0f}" y1="{y0:.0f}" x2="{x1:.0f}" y2="{y1:.0f}" '
                     f'stroke="{EDGE}" stroke-width="1.2"/>')
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2 - 5
            s.append(f'<text x="{mx:.0f}" y="{my:.0f}" fill="{YELLOW}" font-size="11" '
                     f'text-anchor="middle">{lbl}</text>')
            draw(child, cdepth)
        is_leaf = not node.children
        col = GREEN if is_leaf else (BLUE if node is not tree.root else TEXT)
        s.append(f'<circle cx="{x0:.0f}" cy="{y0:.0f}" r="5" fill="{col}"/>')

    draw(tree.root, 0)
    s.append(f'<text x="{ml}" y="{H-18}" fill="{GRAY}" font-size="10">'
             f'Green = leaf (a suffix), blue = internal branch (a repeated substring). '
             f'Every substring is a path from the root.</text>')
    s.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(s))


def _iter_leaves(node):
    if not node.children:
        yield node
        return
    for c in node.children.values():
        yield from _iter_leaves(c)


def main(outdir=None):
    word = "banana"
    t = SuffixTree(word)

    lines = []
    lines.append("Ukkonen suffix tree")
    lines.append("=" * 50)
    lines.append(f'text: "{word}"  (built online in O(n) with suffix links)')
    lines.append("")
    lines.append("substring queries:")
    lines.append(f"{'pattern':>10}{'in text?':>10}{'occurrences':>13}")
    for p in ("ana", "nan", "ban", "na", "xyz", "banana"):
        lines.append(f"{p:>10}{str(t.contains(p)):>10}{t.count_occurrences(p):>13}")
    lines.append("")
    lines.append(f"distinct substrings:      {t.count_distinct_substrings()}")
    lines.append(f"longest repeated substr:  '{t.longest_repeated_substring()}'")
    lines.append("")
    # cross-check against suffix array + LCP
    sa = SA.build_suffix_array(word)
    lcp = SA.build_lcp(word, sa)
    n = len(word)
    distinct_sa = n * (n + 1) // 2 - sum(lcp)
    lines.append(f"cross-check distinct via suffix array + LCP: {distinct_sa}  "
                 f"(matches: {distinct_sa == t.count_distinct_substrings()})")
    lines.append("")
    lines.append("suffixes (leaves), in tree order:")
    suffixes = sorted(word[i:] for i in range(len(word)))
    lines.append("  " + ", ".join(suffixes))

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        _svg(os.path.join(outdir, "suffix_tree.svg"), t, word)

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
