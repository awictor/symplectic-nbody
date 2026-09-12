"""Demo: push-relabel maximum flow -- preflow, heights, and the min cut.

Solves a classic max-flow network, confirms it matches the independent Dinic solver, reports the
per-edge flow and the minimum cut, and draws the network with edges coloured by utilisation and the
min-cut boundary marked.

    python examples/push_relabel_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from push_relabel import PushRelabel  # noqa: E402
import dinic  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Push-relabel maximum flow: move a preflow downhill until it settles\n")

    # CLRS textbook network, nodes 0=s .. 5=t
    edges = [(0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4), (1, 3, 12),
             (3, 2, 9), (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]
    pr = PushRelabel(6)
    idx = [pr.add_edge(u, v, c) for u, v, c in edges]
    flow = pr.max_flow(0, 5)
    dn = dinic.max_flow(6, edges, 0, 5)

    print(f"  6-node network, source=0 sink=5:")
    print(f"    max flow (push-relabel) = {flow}")
    print(f"    max flow (Dinic, independent) = {dn}   {'MATCH' if flow == dn else 'MISMATCH'}\n")

    print("  per-edge flow / capacity:")
    for i, (u, v, c) in enumerate(edges):
        f = pr.flow_on(idx[i])
        bar = "#" * int(12 * f / c) + "." * (12 - int(12 * f / c))
        print(f"    {u}->{v}: {f:2d}/{c:2d}  [{bar}]")

    cut = pr.min_cut(0)
    cut_cap = sum(c for u, v, c in edges if u in cut and v not in cut)
    print(f"\n  minimum cut: source side = {sorted(cut)}")
    print(f"    cut capacity = {cut_cap}  (equals max flow {flow}: max-flow min-cut theorem)\n")

    print("  Push-relabel keeps a PREFLOW where nodes hold excess, and a HEIGHT per node. It PUSHES")
    print("  excess to a lower neighbour, or RELABELS (lifts) a stuck node until it can push. Excess")
    print("  that cannot reach the sink is lifted above the source and drains back; when only the sink")
    print("  holds excess, the preflow has become a maximum flow.")

    _svg(os.path.join(outdir, "push_relabel.svg"), edges, pr, idx, cut)
    print(f"\n  wrote {os.path.join(outdir, 'push_relabel.svg')}")


def _svg(path, edges, pr, idx, cut, width=760, height=420):
    # fixed layout for the 6-node network
    pos = {0: (60, 210), 1: (260, 90), 2: (260, 330), 3: (500, 90), 4: (500, 330), 5: (700, 210)}
    cutset = set(cut)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'Maximum flow network -- edge fill = utilisation, min cut in red</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'green = saturated edge; blue = partial; the source-side cut nodes are ringed red</text>',
    ]

    # edges
    for i, (u, v, c) in enumerate(edges):
        f = pr.flow_on(idx[i])
        (x1, y1), (x2, y2) = pos[u], pos[v]
        frac = f / c if c else 0
        colour = "#06d6a0" if frac >= 0.999 else ("#4dabf7" if frac > 0 else "#30363d")
        # offset slightly for bidirectional pairs
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy) or 1
        ox, oy = -dy / L * 6, dx / L * 6
        parts.append(f'<line x1="{x1+ox:.1f}" y1="{y1+oy:.1f}" x2="{x2+ox:.1f}" y2="{y2+oy:.1f}" '
                     f'stroke="{colour}" stroke-width="{2 + 3*frac:.1f}"/>')
        mx, my = (x1 + x2) / 2 + ox, (y1 + y2) / 2 + oy
        parts.append(f'<text x="{mx:.0f}" y="{my-4:.0f}" fill="#e6edf3" font-size="11" '
                     f'text-anchor="middle">{f}/{c}</text>')

    # nodes
    for node, (x, y) in pos.items():
        ring = "#ff6b6b" if node in cutset else "#8b949e"
        label = {0: "s", 5: "t"}.get(node, str(node))
        parts.append(f'<circle cx="{x}" cy="{y}" r="18" fill="#161b22" stroke="{ring}" '
                     f'stroke-width="3"/>')
        parts.append(f'<text x="{x}" y="{y+5}" fill="#e6edf3" font-size="15" '
                     f'text-anchor="middle">{label}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
