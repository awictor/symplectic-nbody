"""Demo: elementary cellular automata -- Wolfram's four classes.

Prints the rule tables and behaviour class for the famous rules, plus an ASCII space-time view
of rule 90 (the Sierpinski triangle), then draws the space-time diagrams of rules 90, 30, and
110 side by side -- fractal, chaotic, and complex.

    python examples/cellular_automaton_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cellular_automaton import (rule_table, evolve, single_seed_row,  # noqa: E402
                                population)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Elementary cellular automata: 256 rules, each an 8-bit lookup on 3 cells\n")
    print(f"  {'rule':>6}{'table':>12}{'class / note':>34}")
    notes = {0: "1: dies to uniform", 90: "2: Sierpinski fractal (XOR)",
             30: "3: chaotic (used as an RNG)", 110: "4: complex, Turing-complete",
             184: "2: traffic-flow model"}
    for r in (0, 90, 30, 110, 184):
        tbl = "".join(str(b) for b in reversed(rule_table(r)))
        print(f"  {r:>6}{tbl:>12}{notes[r]:>34}")

    print("\n  Rule 90 from a single seed (the Sierpinski triangle):")
    rows = evolve(single_seed_row(61), 90, 20)
    for row in rows:
        print("    " + "".join("#" if c else " " for c in row))

    print("\n  From the simplest imaginable rule -- one output bit per 3-cell neighbourhood --")
    print("  come fractals (90), chaos indistinguishable from random (30, once Mathematica's")
    print("  RNG), and universal computation (110). Complexity needs almost no ingredients.")

    _svg(os.path.join(outdir, "cellular_automaton.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'cellular_automaton.svg')}")


def _svg(path, size=720, pad=40):
    panels = [(90, "Rule 90: Sierpinski", "#4dabf7"),
              (30, "Rule 30: chaos", "#ff6b6b"),
              (110, "Rule 110: complex", "#06d6a0")]
    width = 81
    gens = 80

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">Elementary cellular automata</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'space-time diagrams (time downward) from a single seed: fractal, chaotic, complex</text>',
    ]

    panel_w = (size - 2 * pad) / 3
    cell = min(panel_w * 0.95 / width, (size - 110) / gens)
    top = 60
    for pi, (rule, label, col) in enumerate(panels):
        # rule 110 needs a random-ish start to show its structure; others use single seed
        if rule == 110:
            start = [1 if (i * 7 + 3) % 5 == 0 else 0 for i in range(width)]
        else:
            start = single_seed_row(width)
        rows = evolve(start, rule, gens)
        x0 = pad + panel_w * pi + (panel_w - width * cell) / 2
        for g, row in enumerate(rows):
            for i, c in enumerate(row):
                if c:
                    parts.append(f'<rect x="{x0 + i*cell:.1f}" y="{top + g*cell:.1f}" '
                                 f'width="{cell+0.4:.1f}" height="{cell+0.4:.1f}" fill="{col}"/>')
        parts.append(f'<text x="{x0 + width*cell/2:.1f}" y="{top + gens*cell + 16:.1f}" '
                     f'fill="{col}" font-size="10" text-anchor="middle">{label}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
