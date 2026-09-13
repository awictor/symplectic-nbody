"""Demo: Top Trading Cycles -- the strategy-proof core allocation for swapping indivisible goods.

Reallocates dorm rooms by preference, showing the trading cycles cleared each round, confirms everyone
ends up no worse than they started, and draws the pointing graph with its cycle.

    python examples/top_trading_cycles_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from top_trading_cycles import (top_trading_cycles, is_individually_rational,  # noqa: E402
                                is_pareto_efficient, in_core)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Top Trading Cycles: everyone trades to the core, and no one can game it\n")

    people = ["Ann", "Bo", "Cy", "Di", "Ed", "Fi"]
    n = len(people)
    owner = list(range(n))     # person i owns room i
    # each person's ranking of rooms 0..5 (best first)
    prefs = [
        [2, 1, 0, 3, 4, 5],    # Ann wants Cy's room most
        [2, 0, 1, 4, 3, 5],    # Bo also wants Cy's room
        [0, 2, 1, 3, 4, 5],    # Cy wants Ann's room
        [4, 3, 5, 0, 1, 2],    # Di wants Ed's room
        [3, 4, 5, 0, 1, 2],    # Ed wants Di's room
        [5, 3, 4, 0, 1, 2],    # Fi keeps own
    ]
    alloc, rounds = top_trading_cycles(owner, prefs)

    print(f"  {n} people, each owning one room, each ranking all rooms:\n")
    for r, cycles in enumerate(rounds):
        for cyc in cycles:
            names = " -> ".join(people[p] for p in cyc)
            if len(cyc) == 1:
                print(f"    round {r+1}: {people[cyc[0]]} keeps their own room (self-loop)")
            else:
                print(f"    round {r+1}: trading cycle {names} -> (back to {people[cyc[0]]})")
    print()
    print("  final allocation:")
    for i in range(n):
        moved = "" if alloc[i] == i else "  (traded up)"
        print(f"    {people[i]:>4} gets room {alloc[i]} ({people[alloc[i]]}'s){moved}")
    print()
    print(f"  individually rational (no one worse off): {is_individually_rational(owner, prefs, alloc)}")
    print(f"  Pareto efficient: {is_pareto_efficient(prefs, alloc)}")
    print(f"  in the core (no coalition can do better): {in_core(owner, prefs, alloc)}\n")

    print("  Each person points to the owner of their favourite remaining room; with everyone pointing")
    print("  somewhere, a cycle must exist -- its members trade around it and leave. Repeat on the rest.")
    print("  The result is the unique core allocation, and lying about preferences can never help you --")
    print("  the property that made TTC the basis of kidney-exchange and school-choice mechanisms.")

    _svg(os.path.join(outdir, "top_trading_cycles.svg"), people, owner, prefs)
    print(f"\n  wrote {os.path.join(outdir, 'top_trading_cycles.svg')}")


def _svg(path, people, owner, prefs, width=520, height=520):
    # draw the FIRST-round pointing graph and highlight its cycles
    n = len(people)
    cx, cy, R = width / 2, height / 2 + 10, 180
    pos = {}
    for i in range(n):
        ang = -math.pi / 2 + i * 2 * math.pi / n
        pos[i] = (cx + R * math.cos(ang), cy + R * math.sin(ang))

    # round-1 pointing: i -> owner of i's favourite object (all alive)
    obj_owner = list(owner)
    points = {i: obj_owner[prefs[i][0]] for i in range(n)}
    # find cycle members
    _, rounds = top_trading_cycles(owner, prefs)
    first_cycle_members = set(p for cyc in rounds[0] for p in cyc)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
        f'<path d="M0,0 L7,3 L0,6 Z" fill="#8b949e"/></marker>'
        f'<marker id="arg" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
        f'<path d="M0,0 L7,3 L0,6 Z" fill="#06d6a0"/></marker></defs>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="15">'
        f'TTC round 1: each points to their favourite room\'s owner; cycles clear (green)</text>',
    ]
    # arrows
    for i, j in points.items():
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        if i == j:
            # self-loop
            parts.append(f'<circle cx="{x1:.1f}" cy="{y1-28:.1f}" r="12" fill="none" '
                         f'stroke="#8b949e" stroke-width="1.5" marker-end="url(#ar)"/>')
            continue
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        sx, sy = x1 + ux * 20, y1 + uy * 20
        ex, ey = x2 - ux * 20, y2 - uy * 20
        ingreen = i in first_cycle_members and j in first_cycle_members and points[i] == j
        col = "#06d6a0" if ingreen else "#8b949e"
        mk = "arg" if ingreen else "ar"
        parts.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
                     f'stroke="{col}" stroke-width="{2.5 if ingreen else 1.3}" marker-end="url(#{mk})"/>')
    # nodes
    for i in range(n):
        x, y = pos[i]
        ring = "#06d6a0" if i in first_cycle_members else "#4dabf7"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="20" fill="#161b22" stroke="{ring}" '
                     f'stroke-width="2"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+4:.1f}" fill="#e6edf3" font-size="11" '
                     f'text-anchor="middle">{people[i]}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
