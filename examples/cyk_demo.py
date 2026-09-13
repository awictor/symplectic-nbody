"""Demo: CYK parsing -- context-free recognition by dynamic programming.

Parses balanced-parenthesis strings and an ambiguous grammar, shows the CYK table filling up, counts
parse trees (revealing ambiguity as Catalan numbers), and draws the triangular CYK chart.

    python examples/cyk_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cyk import CNFGrammar, recognize, count_parses, parse_tree, tree_yield, _table  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("CYK parsing: deciding context-free membership in O(n^3), any grammar\n")

    paren = CNFGrammar(
        binary_rules=[("S", "S", "S"), ("S", "L", "R"), ("S", "L", "T"), ("T", "S", "R")],
        terminal_rules=[("L", "("), ("R", ")")],
        start="S",
    )

    print("  Grammar (Chomsky Normal Form) for balanced parentheses:")
    print("    S -> S S | L R | L T,   T -> S R,   L -> '(',   R -> ')'\n")

    tests = ["()", "(())", "()()", "(()())", "(()", ")(", "((("]
    print(f"  {'string':>8}  {'accepted':>9}  {'parses':>7}")
    for s in tests:
        print(f"  {s:>8}  {str(recognize(paren, s)):>9}  {count_parses(paren, s):>7}")

    # show the CYK table for one string
    s = "(())"
    table, n = _table(paren, s)
    print(f"\n  CYK table for '{s}' (cell[i,L] = nonterminals deriving the L-char span at i):")
    for L in range(1, n + 1):
        row = []
        for i in range(0, n - L + 1):
            cell = table[(i, L)]
            row.append("{" + ",".join(sorted(cell)) + "}" if cell else ".")
        print(f"    L={L}: " + "  ".join(f"{c:<8}" for c in row))
    print(f"    -> start symbol S in top cell: string accepted.")

    # ambiguous grammar: parses of a^n = Catalan numbers
    amb = CNFGrammar(binary_rules=[("S", "S", "S")], terminal_rules=[("S", "a")], start="S")
    print("\n  Ambiguous grammar S -> S S | a. Number of parse trees of a^n (Catalan numbers):")
    for n in range(1, 8):
        print(f"    a^{n}: {count_parses(amb, 'a' * n)} parses")

    print("\n  CYK fills a triangular table over all substrings, so it parses ANY context-free")
    print("  grammar -- ambiguous ones included -- which LL/LR parsers cannot.")

    _svg(os.path.join(outdir, "cyk.svg"), paren, "(()())")
    print(f"\n  wrote {os.path.join(outdir, 'cyk.svg')}")


def _svg(path, grammar, s, width=760, height=430):
    table, n = _table(grammar, s)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="15">'
        f'CYK triangular chart for "{s}" -- filled bottom-up, accept if S reaches the apex</text>',
    ]
    cell = min(90, (width - 80) // n)
    x0 = 40
    y0 = 360
    for L in range(1, n + 1):
        for i in range(0, n - L + 1):
            nts = table[(i, L)]
            x = x0 + (i + (L - 1) / 2) * cell
            y = y0 - (L - 1) * 52
            has_start = grammar.start in nts
            col = "#06d6a0" if has_start else ("#4dabf7" if nts else "#30363d")
            parts.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{cell-8:.0f}" height="44" rx="4" '
                         f'fill="#161b22" stroke="{col}" stroke-width="{2.5 if has_start else 1.3}"/>')
            label = ",".join(sorted(nts)) if nts else "-"
            parts.append(f'<text x="{x + (cell-8)/2:.0f}" y="{y+27:.0f}" fill="{col}" '
                         f'font-size="11" text-anchor="middle">{label}</text>')
    # the input row
    for i, ch in enumerate(s):
        x = x0 + i * cell
        parts.append(f'<text x="{x + (cell-8)/2:.0f}" y="{y0+62:.0f}" fill="#ffd43b" '
                     f'font-size="16" text-anchor="middle">{ch}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
