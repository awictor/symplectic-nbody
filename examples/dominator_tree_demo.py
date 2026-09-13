"""Demo: dominator tree of a control-flow graph.

Takes a small control-flow graph (an if/else with a loop), computes its immediate dominators, and
draws the CFG beside its dominator tree.

    python examples/dominator_tree_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dominator_tree import DominatorTree  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    # A control-flow graph resembling:
    #   0 entry
    #   0 -> 1 (test)
    #   1 -> 2 (then), 1 -> 3 (else)
    #   2 -> 4, 3 -> 4 (join)
    #   4 -> 5 (loop body), 5 -> 4 (back edge), 4 -> 6 (exit test), 6 exit
    labels = ["entry", "test", "then", "else", "join", "body", "exit"]
    edges = [
        (0, 1),
        (1, 2), (1, 3),
        (2, 4), (3, 4),
        (4, 5), (5, 4),
        (4, 6),
    ]
    n = len(labels)
    dt = DominatorTree(n, edges, entry=0)
    idom = dt.immediate_dominators()

    print("Dominator tree: who must you pass through to reach each block\n")
    print("  Control-flow graph (if/else joining into a loop):")
    for u, v in edges:
        print(f"    {labels[u]:>5} -> {labels[v]}")

    print("\n  Immediate dominators:")
    for v in range(n):
        if v in idom:
            d = idom[v]
            tag = "(entry)" if v == 0 else f"idom = {labels[d]}"
            print(f"    {labels[v]:>5}: {tag}")

    print("\n  Full dominators of each block (every entry path passes through these):")
    for v in range(n):
        if v in idom:
            doms = sorted(dt.dominators_of(v))
            print(f"    {labels[v]:>5}: {', '.join(labels[d] for d in doms)}")

    print("\n  Note: 'join' is dominated only by entry/test -- neither branch dominates it (both")
    print("  reach it). 'body' and 'exit' are dominated by 'join'. This is exactly the structure")
    print("  SSA construction uses to place phi-functions and to find the natural loop 4<->5.")

    _svg(os.path.join(outdir, "dominator_tree.svg"), labels, edges, dt)
    print(f"\n  wrote {os.path.join(outdir, 'dominator_tree.svg')}")


def _svg(path, labels, edges, dt, width=760, height=460):
    n = len(labels)
    idom = dt.immediate_dominators()

    # ---- left: CFG, hand-placed layered layout ------------------------------------------
    cfg_pos = {
        0: (150, 60),
        1: (150, 130),
        2: (90, 210), 3: (210, 210),
        4: (150, 290),
        5: (60, 290),
        6: (150, 400),
    }

    # ---- right: dominator tree by BFS depth ---------------------------------------------
    children = {v: [] for v in idom}
    for v in idom:
        if v != dt.entry:
            children[idom[v]].append(v)
    depth = {dt.entry: 0}
    order = [dt.entry]
    q = [dt.entry]
    while q:
        u = q.pop(0)
        for c in sorted(children[u]):
            depth[c] = depth[u] + 1
            order.append(c)
            q.append(c)
    layers = {}
    for v in order:
        layers.setdefault(depth[v], []).append(v)
    maxd = max(layers) if layers else 0
    tpos = {}
    ox = 430
    for d in range(maxd + 1):
        row = layers[d]
        for j, v in enumerate(row):
            x = ox + (width - ox) * (j + 1) / (len(row) + 1)
            y = 90 + d * (height - 150) / max(1, maxd)
            tpos[v] = (x, y)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="30" fill="#e6edf3" font-size="15">'
        'Control-flow graph (left) and its dominator tree (right)</text>',
        '<text x="150" y="52" fill="#8b949e" font-size="11" text-anchor="middle">CFG</text>',
        f'<text x="{(ox+width)//2}" y="80" fill="#8b949e" font-size="11" '
        f'text-anchor="middle">dominator tree</text>',
        '<defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">'
        '<path d="M0,0 L7,3 L0,6 Z" fill="#8b949e"/></marker></defs>',
    ]

    def arrow(x1, y1, x2, y2, col, mk=True):
        import math
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy) or 1
        ux, uy = dx / d, dy / d
        r = 17
        sx, sy = x1 + ux * r, y1 + uy * r
        ex, ey = x2 - ux * r, y2 - uy * r
        m = ' marker-end="url(#a)"' if mk else ""
        return (f'<line x1="{sx:.0f}" y1="{sy:.0f}" x2="{ex:.0f}" y2="{ey:.0f}" '
                f'stroke="{col}" stroke-width="1.6"{m}/>')

    for u, v in edges:
        x1, y1 = cfg_pos[u]
        x2, y2 = cfg_pos[v]
        parts.append(arrow(x1, y1, x2, y2, "#30363d"))
    for v in range(n):
        x, y = cfg_pos[v]
        parts.append(f'<circle cx="{x}" cy="{y}" r="16" fill="#161b22" stroke="#4dabf7" '
                     f'stroke-width="2"/>')
        parts.append(f'<text x="{x}" y="{y+3}" fill="#4dabf7" font-size="8" '
                     f'text-anchor="middle">{labels[v]}</text>')

    for v in tpos:
        if v == dt.entry:
            continue
        x1, y1 = tpos[idom[v]]
        x2, y2 = tpos[v]
        parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                     f'stroke="#06d6a0" stroke-width="2"/>')
    for v in tpos:
        x, y = tpos[v]
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="16" fill="#161b22" stroke="#06d6a0" '
                     f'stroke-width="2"/>')
        parts.append(f'<text x="{x:.0f}" y="{y+3:.0f}" fill="#06d6a0" font-size="8" '
                     f'text-anchor="middle">{labels[v]}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
