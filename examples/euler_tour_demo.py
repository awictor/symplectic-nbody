"""Euler tour demo: flatten a tree, show subtrees as contiguous intervals of tour time (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import euler_tour as ET


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
YELLOW = "#ffd43b"


def main(outdir=None):
    # a small tree
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (4, 7), (2, 5), (2, 6)]
    n = 8
    t = ET.euler_tour(n, edges, root=0)

    lines = []
    lines.append("Euler tour: flattening a tree so subtrees become array ranges")
    lines.append("=" * 62)
    lines.append(f"{n}-node tree, rooted at 0")
    lines.append("")
    lines.append(f"{'node':>5}{'tin':>5}{'tout':>6}{'depth':>7}{'subtree range':>16}{'size':>6}")
    for v in range(n):
        lo, hi = ET.subtree_range(t, v)
        lines.append(f"{v:>5}{t['tin'][v]:>5}{t['tout'][v]:>6}{t['depth'][v]:>7}"
                     f"{f'[{lo}, {hi}]':>16}{ET.subtree_size(t, v):>6}")
    lines.append("")
    lines.append("tour order (nodes by entry time): " + " ".join(str(x) for x in t["order"]))
    lines.append("")
    lines.append("queries that become O(1) or O(log n) on the flattened array:")
    lines.append(f"  is 1 an ancestor of 7?  {ET.is_ancestor(t, 1, 7)}  (tin/tout containment)")
    lines.append(f"  is 2 an ancestor of 7?  {ET.is_ancestor(t, 2, 7)}")
    lines.append(f"  subtree of 1 = {sorted(ET.subtree_nodes(t, 1))}")
    vals = [1, 2, 3, 4, 5, 6, 7, 8]
    lines.append(f"  subtree_sum(1) with values 1..8 = {ET.subtree_sum(t, vals, 1):.0f}")
    lines.append("")
    lines.append("Every subtree is a contiguous slice of tour time, so subtree sums/updates")
    lines.append("reduce to range operations on an array (pair with a Fenwick tree for O(log n)).")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W, H = 720, 420
        # top: the tree drawn by depth/position; bottom: the tour timeline with subtree bracket
        # position nodes: x by tin, y by depth
        ml, mt = 40, 60
        col_w = (W - 2 * ml) / n
        row_h = 60

        def nx(v):
            return ml + (t["tin"][v] + 0.5) * col_w

        def ny(v):
            return mt + t["depth"][v] * row_h

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'Euler tour: the subtree of node 1 is one contiguous span of tour time</text>')
        # highlight subtree of node 1 span on the timeline
        target = 1
        lo, hi = ET.subtree_range(t, target)
        # tree edges
        for (u, v) in edges:
            s.append(f'<line x1="{nx(u):.1f}" y1="{ny(u):.1f}" x2="{nx(v):.1f}" y2="{ny(v):.1f}" '
                     f'stroke="{GRAY}" stroke-width="1.2"/>')
        # nodes, subtree of target in green
        subset = set(ET.subtree_nodes(t, target))
        for v in range(n):
            col = GREEN if v in subset else BLUE
            s.append(f'<circle cx="{nx(v):.1f}" cy="{ny(v):.1f}" r="13" fill="{col}" '
                     f'stroke="{BG}" stroke-width="2"/>')
            s.append(f'<text x="{nx(v):.1f}" y="{ny(v)+4:.1f}" fill="{BG}" font-size="11" '
                     f'text-anchor="middle" font-weight="bold">{v}</text>')
        # timeline at the bottom
        ty = H - 90
        s.append(f'<text x="{ml}" y="{ty-14}" fill="{GRAY}" font-size="11">tour time (entry order)</text>')
        for i in range(n):
            x = ml + (i + 0.5) * col_w
            node = t["order"][i]
            inspan = lo <= i <= hi
            col = GREEN if inspan else "#161b22"
            s.append(f'<rect x="{x-col_w/2+2:.1f}" y="{ty}" width="{col_w-4:.1f}" height="30" '
                     f'fill="{col}" fill-opacity="0.4" stroke="{GRAY}" stroke-width="0.5"/>')
            s.append(f'<text x="{x:.1f}" y="{ty+20:.1f}" fill="{TEXT}" font-size="11" '
                     f'text-anchor="middle">{node}</text>')
        # bracket over the subtree span
        x0 = ml + lo * col_w + 2
        x1 = ml + (hi + 1) * col_w - 2
        s.append(f'<line x1="{x0:.1f}" y1="{ty+38}" x2="{x1:.1f}" y2="{ty+38}" '
                 f'stroke="{GREEN}" stroke-width="2"/>')
        s.append(f'<text x="{(x0+x1)/2:.1f}" y="{ty+52:.1f}" fill="{GREEN}" font-size="10" '
                 f'text-anchor="middle">subtree of node 1 = tour[{lo}..{hi}]</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "euler_tour.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
