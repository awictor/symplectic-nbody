"""Demo: DPLL SAT solving -- deciding Boolean formulas and proving the pigeonhole principle.

Solves a small satisfiable formula, shows the pigeonhole principle is unsatisfiable, and draws the
DPLL search as the shrinking clause set under unit propagation.

    python examples/dpll_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dpll import solve, is_satisfied, pigeonhole  # noqa: E402


def lit_str(lit):
    return f"x{lit}" if lit > 0 else f"~x{-lit}"


def clause_str(clause):
    return "(" + " v ".join(lit_str(l) for l in clause) + ")"


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("DPLL: Boolean satisfiability by backtracking with propagation\n")

    # a satisfiable formula
    formula = [[1, 2, -3], [-1, 3], [-2, 3], [1, -2], [2, 3]]
    print("  Formula (CNF, AND of ORs):")
    print("    " + " ^ ".join(clause_str(c) for c in formula))
    sat, model = solve(formula)
    print(f"\n  DPLL: {'SATISFIABLE' if sat else 'UNSATISFIABLE'}")
    if sat:
        assign = ", ".join(f"x{v}={'T' if model[v] else 'F'}" for v in sorted(model))
        print(f"    model: {assign}")
        print(f"    verified: {is_satisfied(formula, model)}\n")

    # the pigeonhole principle: n+1 pigeons cannot fit in n holes with no sharing
    print("  Pigeonhole principle -- can n+1 pigeons occupy n holes, no two sharing?")
    for n in range(1, 5):
        clauses, nv = pigeonhole(n)
        sat, _ = solve(clauses, nv)
        print(f"    {n+1} pigeons, {n} holes: {nv} vars, {len(clauses)} clauses -> "
              f"{'SAT' if sat else 'UNSAT (proven impossible)'}")
    print("    Every case UNSAT -- DPLL proves the combinatorial impossibility by exhausting the")
    print("    (heavily pruned) search tree, never enumerating all 2^n assignments.\n")

    # trace clause-count reduction under unit propagation on a forcing chain
    chain = [[1], [-1, 2], [-2, 3], [-3, 4], [-4, 5]]
    print("  Unit propagation on a forcing chain x1 -> x2 -> ... -> x5:")
    sat, model = solve(chain)
    vals = " ".join(f"x{v}={'T' if model[v] else 'F'}" for v in sorted(model))
    print(f"    {vals}  -- one forced unit cascades through the whole chain.")

    _svg(os.path.join(outdir, "dpll.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'dpll.svg')}")


def _svg(path, width=760, height=430):
    # show clause count of the pigeonhole family growing while all stay UNSAT,
    # and the DPLL vs brute-force search-space comparison
    ns = [1, 2, 3, 4, 5]
    clause_counts = []
    brute_space = []
    for n in ns:
        clauses, nv = pigeonhole(n)
        clause_counts.append(len(clauses))
        brute_space.append(nv)          # log2 of the brute search space (2^nv)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Pigeonhole SAT instances: all UNSAT, and the search space DPLL avoids brute-forcing</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'yellow: number of clauses. red: log2 of the 2^n brute-force assignment space DPLL prunes'
        f'</text>',
    ]

    ox, oy = 90, 350
    bw = 90
    gap = 40
    max_clauses = max(clause_counts)
    max_log = max(brute_space)
    scale_c = 230 / max_clauses
    scale_b = 230 / max_log

    for i, n in enumerate(ns):
        x = ox + i * (bw + gap)
        # clause bar (yellow)
        hc = clause_counts[i] * scale_c
        parts.append(f'<rect x="{x}" y="{oy-hc:.1f}" width="{bw/2-2:.0f}" height="{hc:.1f}" '
                     f'fill="#ffd43b"/>')
        parts.append(f'<text x="{x+bw/4:.0f}" y="{oy-hc-6:.1f}" fill="#ffd43b" font-size="11" '
                     f'text-anchor="middle">{clause_counts[i]}</text>')
        # brute log-space bar (red)
        hb = brute_space[i] * scale_b
        parts.append(f'<rect x="{x+bw/2:.0f}" y="{oy-hb:.1f}" width="{bw/2-2:.0f}" '
                     f'height="{hb:.1f}" fill="#ff6b6b"/>')
        parts.append(f'<text x="{x+3*bw/4:.0f}" y="{oy-hb-6:.1f}" fill="#ff6b6b" font-size="11" '
                     f'text-anchor="middle">2^{brute_space[i]}</text>')
        parts.append(f'<text x="{x+bw/2:.0f}" y="{oy+20}" fill="#e6edf3" font-size="12" '
                     f'text-anchor="middle">{n+1}p/{n}h</text>')
        parts.append(f'<text x="{x+bw/2:.0f}" y="{oy+38}" fill="#06d6a0" font-size="11" '
                     f'text-anchor="middle">UNSAT</text>')

    parts.append(f'<line x1="{ox-10}" y1="{oy}" x2="{width-20}" y2="{oy}" '
                 f'stroke="#30363d" stroke-width="1"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
