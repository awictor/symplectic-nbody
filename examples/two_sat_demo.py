"""Demo: 2-SAT solved in linear time via the implication graph and SCCs.

Solves a small scheduling-style 2-SAT instance, shows the canonical unsatisfiable formula, and
confirms the verdict against brute force. Draws the implication graph with each SCC colored.

    python examples/two_sat_demo.py [output_dir]
"""

import itertools
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from two_sat import TwoSAT  # noqa: E402
from tarjan_scc import strongly_connected_components  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("2-SAT: satisfying two-literal clauses in linear time via SCCs\n")

    # a small constraint problem: 3 features, each on/off, with either/or requirements
    # (F1 or F2): at least one of features 1,2
    # (not F1 or F3): if F1 then F3
    # (not F2 or not F3): not both F2 and F3
    s = TwoSAT(3)
    clauses = [(1, 2), (-1, 3), (-2, -3)]
    for c in clauses:
        s.add_clause(*c)

    print("  Formula: (F1 v F2) AND (~F1 v F3) AND (~F2 v ~F3)")
    a = s.solve()
    if a:
        print(f"  SATISFIABLE. One assignment: "
              + ", ".join(f"F{i+1}={'T' if a[i] else 'F'}" for i in range(3)))
        print(f"  verifies against every clause: {s.check(a)}")

    # brute-force confirmation
    def brute(n, cls):
        for bits in itertools.product([False, True], repeat=n):
            if all((bits[abs(x)-1] if x > 0 else not bits[abs(x)-1]) or
                   (bits[abs(y)-1] if y > 0 else not bits[abs(y)-1]) for x, y in cls):
                return True
        return False
    print(f"  brute force agrees it is satisfiable: {brute(3, clauses)}")

    # unsatisfiable example
    print("\n  Unsatisfiable formula: (a v b)(a v ~b)(~a v b)(~a v ~b)")
    u = TwoSAT(2)
    for c in [(1, 2), (1, -2), (-1, 2), (-1, -2)]:
        u.add_clause(*c)
    print(f"  solver verdict: {'UNSATISFIABLE' if u.solve() is None else 'satisfiable'}")
    print(f"  reason: some variable and its negation land in the same SCC of the implication graph")

    print("\n  Each clause (a v b) becomes two implications (~a -> b) and (~b -> a). The formula is")
    print("  satisfiable iff no variable shares a strongly connected component with its own negation;")
    print("  when it does, x -> ~x -> x forces a contradiction. Assignment reads off the SCC order.")

    _svg(os.path.join(outdir, "two_sat.svg"), 3, clauses)
    print(f"\n  wrote {os.path.join(outdir, 'two_sat.svg')}")


def _svg(path, n, clauses, width=760, height=430):
    colors = ["#4dabf7", "#ffd43b", "#ff6b6b", "#06d6a0", "#b197fc", "#ff922b"]

    # build the implication graph the same way TwoSAT does
    def vertex(lit):
        var = abs(lit) - 1
        return 2 * var if lit > 0 else 2 * var + 1
    edges = []
    for a, b in clauses:
        va, vb = vertex(a), vertex(b)
        edges.append((va ^ 1, vb))
        edges.append((vb ^ 1, va))
    comps = strongly_connected_components(2 * n, edges)
    comp_of = [0] * (2 * n)
    for cid, comp in enumerate(comps):
        for v in comp:
            comp_of[v] = cid

    # layout: two rows -- true literals on top, negations on bottom
    labels = {}
    pos = {}
    for i in range(n):
        labels[2 * i] = f"F{i+1}"
        labels[2 * i + 1] = f"~F{i+1}"
        x = 120 + i * 200
        pos[2 * i] = (x, 150)
        pos[2 * i + 1] = (x, 320)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'2-SAT implication graph (nodes colored by SCC)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'top row = literals true, bottom = negations; edges are clause implications</text>',
        '<defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="3" '
        'orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#484f58"/></marker></defs>',
    ]
    for (u, v) in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy) or 1
        ax, ay = x0 + dx / L * 22, y0 + dy / L * 22
        bx, by = x1 - dx / L * 26, y1 - dy / L * 26
        mx, my = (x0 + x1) / 2 - dy / L * 18, (y0 + y1) / 2 + dx / L * 18
        parts.append(f'<path d="M{ax:.1f},{ay:.1f} Q{mx:.1f},{my:.1f} {bx:.1f},{by:.1f}" '
                     f'fill="none" stroke="#484f58" stroke-width="1.3" marker-end="url(#a)"/>')
    for v, (x, y) in pos.items():
        col = colors[comp_of[v] % len(colors)]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="22" fill="{col}" '
                     f'stroke="#0d1117" stroke-width="2"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+5:.1f}" fill="#0d1117" font-size="13" '
                     f'text-anchor="middle" font-weight="bold">{labels[v]}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
