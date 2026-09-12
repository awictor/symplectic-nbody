"""Demo: minimum-cost maximum flow -- ship the most goods for the least money.

Builds a small shipping network (two factories -> two warehouses -> two stores) where every pipe has
a capacity and a per-unit shipping cost, finds the maximum flow at minimum total cost, and draws the
network with each edge labelled by the flow it carries.

    python examples/min_cost_flow_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from min_cost_flow import MinCostFlow, assignment_min_cost  # noqa: E402

# nodes: 0=source, 1=factory A, 2=factory B, 3=warehouse X, 4=warehouse Y, 5=store, sink=6
NAMES = ["source", "factA", "factB", "whX", "whY", "hub", "sink"]
# (u, v, capacity, cost per unit)
EDGES = [
    (0, 1, 4, 0), (0, 2, 4, 0),        # supply from each factory
    (1, 3, 3, 2), (1, 4, 2, 4),        # factory A -> warehouses
    (2, 3, 2, 3), (2, 4, 3, 1),        # factory B -> warehouses
    (3, 5, 4, 1), (4, 5, 4, 2),        # warehouses -> hub
    (5, 6, 7, 0),                      # hub -> sink (demand)
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    mcf = MinCostFlow(7)
    # remember the forward edge slot so we can read the flow back
    slots = []
    for u, v, cap, cost in EDGES:
        slots.append((u, len(mcf.graph[u]), cap))
        mcf.add_edge(u, v, cap, cost)

    flow, total = mcf.min_cost_max_flow(0, 6)

    print("Minimum-cost maximum flow: a factory-to-store shipping network\n")
    print(f"  Maximum shippable: {flow} units, at minimum total cost {total}\n")
    print("  Flow on each pipe (used / capacity, unit cost):")
    edge_flow = {}
    for (u, v, cap, cost), (su, si, scap) in zip(EDGES, slots):
        used = scap - mcf.graph[su][si][1]        # capacity consumed = original - residual
        edge_flow[(u, v)] = used
        if u == 0 or v == 6:
            continue
        print(f"    {NAMES[u]:6s} -> {NAMES[v]:6s}  {used}/{cap}  @ {cost}/unit  = {used*cost}")

    print("\n  Successive shortest paths: each round routes flow along the cheapest remaining path in")
    print("  the residual graph (with negative-cost reverse arcs handled by SPFA), so the running")
    print("  cost is always minimal for the flow sent. When no path remains the flow is both maximum")
    print("  and cheapest -- generalising plain max-flow (zero costs) and the assignment problem.")

    # bonus: the same solver as an assignment problem
    cost_matrix = [[9, 2, 7], [6, 4, 3], [5, 8, 1]]
    acost, assign = assignment_min_cost(cost_matrix)
    print(f"\n  Same engine, assignment problem: 3 workers -> 3 jobs, min cost {acost}")
    print(f"    worker->job: {assign}  (worker i takes job assign[i])")

    _svg(os.path.join(outdir, "min_cost_flow.svg"), edge_flow)
    print(f"\n  wrote {os.path.join(outdir, 'min_cost_flow.svg')}")


def _svg(path, edge_flow, width=760, height=430):
    pos = {
        0: (60, 215), 1: (210, 120), 2: (210, 310),
        3: (400, 120), 4: (400, 310), 5: (560, 215), 6: (700, 215),
    }
    caps = {(u, v): cap for u, v, cap, _ in EDGES}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Min-cost max-flow: pipe thickness = flow carried, label = used/capacity</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'the cheapest way to push the maximum flow from source (left) to sink (right)</text>',
    ]

    for (u, v), cap in caps.items():
        used = edge_flow.get((u, v), 0)
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        if used > 0:
            col, w = "#06d6a0", 1.5 + 2.0 * used
        else:
            col, w = "#8b949e", 1.0
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
                     f'stroke-width="{w:.1f}" opacity="0.85"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 6
        parts.append(f'<text x="{mx:.0f}" y="{my:.0f}" fill="#ffd43b" font-size="11" '
                     f'text-anchor="middle">{used}/{cap}</text>')

    for v, (x, y) in pos.items():
        parts.append(f'<circle cx="{x}" cy="{y}" r="22" fill="#161b22" '
                     f'stroke="#4dabf7" stroke-width="2"/>')
        parts.append(f'<text x="{x}" y="{y+4}" fill="#e6edf3" font-size="11" '
                     f'text-anchor="middle">{NAMES[v]}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
