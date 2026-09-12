"""Demo: the Steiner tree -- cheapest network connecting terminals, routing through junctions.

Connects a set of terminal cities at least cost, allowing the tree to route through optional junction
towns (Steiner points) when cheaper, and contrasts it with the terminal-only minimum spanning tree.
Draws the graph with terminals, the chosen Steiner point, and the difference highlighted.

    python examples/steiner_tree_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from steiner_tree import steiner_tree, _mst_weight_on, brute_steiner_tree  # noqa: E402

# four terminal cities at the corners; a hub in the middle links them cheaply
NAMES = ["NW", "NE", "SE", "SW", "hub"]
EDGES = [
    (0, 1, 10), (1, 2, 10), (2, 3, 10), (3, 0, 10),   # expensive perimeter
    (0, 4, 3), (1, 4, 3), (2, 4, 3), (3, 4, 3),        # cheap spokes to the hub
]
TERMINALS = [0, 1, 2, 3]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    st = steiner_tree(5, EDGES, TERMINALS)
    mst_terminals = _mst_weight_on(set(TERMINALS), EDGES)

    print("Steiner tree: cheapest network connecting terminals via optional junctions\n")
    print(f"  terminals: {[NAMES[t] for t in TERMINALS]}")
    print(f"  a hub vertex '{NAMES[4]}' is available as a Steiner point (not required)\n")

    print(f"  terminal-only MST (perimeter, no hub): {mst_terminals}")
    print(f"  Steiner tree (allowed to use the hub):  {st}")
    print(f"  saving from routing through the hub:    {mst_terminals - st}\n")

    print(f"  verified against brute force: {st == brute_steiner_tree(5, EDGES, TERMINALS)}")

    # show a few terminal subsets
    print("\n  Steiner cost for various terminal sets:")
    for subset in [[0, 2], [0, 1, 2], [0, 1, 2, 3]]:
        w = steiner_tree(5, EDGES, subset)
        print(f"    {[NAMES[t] for t in subset]}: {w}")

    print("\n  The Dreyfus-Wagner DP fills dp[S][v] = cheapest tree connecting terminal set S and")
    print("  reaching vertex v, by MERGING disjoint terminal subsets at a shared root and GROWING")
    print("  along shortest paths. Exponential only in the number of terminals -- so many terminals")
    print("  in a large graph stay tractable, unlike brute-forcing every subset of Steiner points.")

    _svg(os.path.join(outdir, "steiner_tree.svg"), st, mst_terminals)
    print(f"\n  wrote {os.path.join(outdir, 'steiner_tree.svg')}")


def _svg(path, st, mst_terminals, width=720, height=420):
    pos = {0: (170, 110), 1: (540, 110), 2: (540, 340), 3: (170, 340), 4: (355, 225)}
    # the Steiner solution here is the four spokes to the hub
    steiner_edges = {(0, 4), (1, 4), (2, 4), (3, 4)}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Steiner tree: connect the 4 corner terminals through the cheap hub</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'green = chosen Steiner tree (cost {st}); grey = unused edges; '
        f'perimeter-only MST would cost {mst_terminals}</text>',
    ]

    # all edges
    for u, v, w in EDGES:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        key = (min(u, v), max(u, v))
        chosen = key in {(min(a, b), max(a, b)) for a, b in steiner_edges}
        col = "#06d6a0" if chosen else "#484f58"
        wid = 3.5 if chosen else 1.2
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
                     f'stroke-width="{wid}"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        parts.append(f'<text x="{mx:.0f}" y="{my-3:.0f}" fill="{col}" font-size="11" '
                     f'text-anchor="middle">{w}</text>')

    # nodes
    for v, (x, y) in pos.items():
        if v == 4:
            fill, label = "#ffd43b", "hub"
        else:
            fill, label = "#4dabf7", NAMES[v]
        parts.append(f'<circle cx="{x}" cy="{y}" r="22" fill="{fill}"/>')
        parts.append(f'<text x="{x}" y="{y+5}" fill="#0d1117" font-size="13" '
                     f'text-anchor="middle" font-weight="bold">{label}</text>')

    parts.append(f'<text x="{pos[4][0]}" y="{pos[4][1]+40}" fill="#ffd43b" font-size="11" '
                 f'text-anchor="middle">Steiner point</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
