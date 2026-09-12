"""Demo: WalkSAT -- solving satisfiability by randomized local search.

Solves random 3-SAT formulas by flipping variables to repair conflicts, tracks how the unsatisfied-
clause count falls during the search, and contrasts its incompleteness with complete DPLL. Draws the
conflict-count trajectory.

    python examples/walksat_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from walksat import solve, is_satisfied, dpll_satisfiable, _unsatisfied, _LCG  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("WalkSAT: solving SAT by flipping variables to repair conflicts\n")

    c = [[1, 2, -3], [-1, 3], [-2, 3], [1, -2], [2, 3]]
    sat, model = solve(c)
    print("  small formula (x1 v x2 v ~x3) ^ (~x1 v x3) ^ ...:")
    assign = ", ".join(f"x{v}={'T' if model[v] else 'F'}" for v in sorted(model))
    print(f"    solved: {sat}, model: {assign}, verified: {is_satisfied(c, model)}\n")

    # random 3-SAT near-but-below the phase transition (ratio ~4)
    rng = _LCG(42)
    n = 40
    m = 160
    clauses = []
    for _ in range(m):
        clauses.append([((rng.next() % n) + 1) * (1 if rng.next() % 2 else -1) for _ in range(3)])
    is_sat = dpll_satisfiable(clauses, n)
    sat, model = solve(clauses, n, max_flips=5000, max_tries=20)
    print(f"  random 3-SAT: {n} variables, {m} clauses (ratio {m/n:.1f})")
    print(f"    DPLL says satisfiable: {is_sat}")
    print(f"    WalkSAT found a model: {sat}"
          f"{' (verified)' if sat and is_satisfied(clauses, model) else ''}")

    # trace the conflict count during one solve
    trace = _trace_conflicts(clauses, n, max_flips=1200)
    print(f"\n  conflict trajectory: started with {trace[0]} unsatisfied clauses, "
          f"reached {min(trace)} at best")
    print("    each flip picks a random unsatisfied clause and flips one of its variables -- greedily")
    print("    (fewest clauses broken) most of the time, randomly sometimes to escape local minima.")

    print("\n  WalkSAT is INCOMPLETE: it can't prove unsatisfiability, only find models. But on large")
    print("  satisfiable instances -- where DPLL's backtracking tree explodes -- it often lands a")
    print("  solution almost instantly, which is why stochastic local search underpins the fastest")
    print("  incomplete SAT solvers.")

    _svg(os.path.join(outdir, "walksat.svg"), trace)
    print(f"\n  wrote {os.path.join(outdir, 'walksat.svg')}")


def _trace_conflicts(clauses, n, max_flips):
    """Run a single WalkSAT try, recording the number of unsatisfied clauses after each flip."""
    rng = _LCG(7)
    var_clauses = {v: [] for v in range(1, n + 1)}
    for i, cl in enumerate(clauses):
        for l in cl:
            var_clauses[abs(l)].append(i)
    assignment = {v: (rng.next() % 2 == 0) for v in range(1, n + 1)}
    from walksat import _break_count
    trace = []
    for _ in range(max_flips):
        unsat = _unsatisfied(clauses, assignment)
        trace.append(len(unsat))
        if not unsat:
            break
        clause = clauses[rng.choice(unsat)]
        vars_in = [abs(l) for l in clause]
        if rng.rand_float() < 0.5:
            v = rng.choice(vars_in)
        else:
            v = min(vars_in, key=lambda x: _break_count(clauses, var_clauses, assignment, x))
        assignment[v] = not assignment[v]
    return trace


def _svg(path, trace, width=760, height=360):
    n = len(trace)
    ox, oy = 60, 300
    pw, ph = width - 100, 240
    ymax = max(trace) if trace else 1

    def px(i):
        return ox + i / max(1, n - 1) * pw

    def py(v):
        return oy - v / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'WalkSAT conflict trajectory: unsatisfied clauses vs flips</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'the count jitters down as flips repair conflicts, reaching 0 when a model is found</text>',
    ]
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#30363d"/>')
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(trace))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="1.3"/>')
    if trace and trace[-1] == 0:
        parts.append(f'<circle cx="{px(n-1):.1f}" cy="{py(0):.1f}" r="5" fill="#06d6a0"/>')
        parts.append(f'<text x="{px(n-1)-6:.0f}" y="{py(0)-8:.0f}" fill="#06d6a0" font-size="11" '
                     f'text-anchor="end">solved</text>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+30}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">flips -></text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
