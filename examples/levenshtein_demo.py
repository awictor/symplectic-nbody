"""Demo: Levenshtein edit distance -- the DP table, alignment, and spell-check.

Shows the edit distance and the actual alignment for a classic pair, then a spell-checker
ranking candidate words by distance. Draws the dynamic-programming table as a heatmap with the
backtrace path that spells out the minimal edits.

    python examples/levenshtein_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from levenshtein import (distance, alignment, apply_ops, similarity,  # noqa: E402
                         damerau_distance, _table)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    a, b = "kitten", "sitting"
    print("Levenshtein edit distance: fewest insert/delete/substitute edits\n")
    print(f"  '{a}' -> '{b}' : distance {distance(a, b)}, similarity {similarity(a, b):.2f}")
    print("  alignment:")
    for op in alignment(a, b):
        if op[0] == "match":
            print(f"    keep       {op[1]}")
        elif op[0] == "substitute":
            print(f"    substitute {op[1]} -> {op[2]}")
        elif op[0] == "delete":
            print(f"    delete     {op[1]}")
        else:
            print(f"    insert     {op[1]}")
    print(f"  applying the ops reproduces '{apply_ops(a, alignment(a, b))}'\n")

    # spell-checker: rank a dictionary by distance to a misspelling
    typo = "recieve"
    words = ["receive", "believe", "relieve", "receipt", "deceive", "recede", "receiver"]
    ranked = sorted(words, key=lambda w: (distance(typo, w), w))
    print(f"  spell-check '{typo}' against a small dictionary (nearest first):")
    print(f"  {'word':>10}{'distance':>10}{'damerau':>9}")
    for w in ranked:
        print(f"  {w:>10}{distance(typo, w):>10}{damerau_distance(typo, w):>9}")
    print(f"\n  best guess: '{ranked[0]}' -- and Damerau sees 'recieve'->'receive' as a single")
    print("  adjacent-swap typo (distance 1), the commonest kind of mistake. The same DP powers")
    print("  fuzzy search, diff tools, and DNA sequence alignment.")

    _svg(os.path.join(outdir, "levenshtein.svg"), a, b)
    print(f"\n  wrote {os.path.join(outdir, 'levenshtein.svg')}")


def _svg(path, a, b, w=760, h=430):
    d = _table(a, b)
    m, n = len(a), len(b)
    dmax = max(max(row) for row in d) or 1

    # backtrace path cells (set of (i,j) visited)
    trace = set()
    i, j = m, n
    while i > 0 or j > 0:
        trace.add((i, j))
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    trace.add((0, 0))

    cell = 42
    x0, y0 = 90, 90
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Levenshtein DP table: cost to reach each corner</text>',
        f'<text x="20" y="46" fill="#8b949e" font-size="12">'
        f'cell shade = edit cost of the two prefixes; green trace = the minimal-edit alignment</text>',
    ]

    # column headers (b) and row headers (a)
    parts.append(f'<text x="{x0 - cell // 2:.1f}" y="{y0 - cell - 6:.1f}" fill="#8b949e" '
                 f'font-size="12" text-anchor="middle"></text>')
    for j in range(n + 1):
        label = "" if j == 0 else b[j - 1]
        parts.append(f'<text x="{x0 + j * cell + cell / 2:.1f}" y="{y0 - cell - 6:.1f}" '
                     f'fill="#ffd43b" font-size="13" text-anchor="middle">{label}</text>')
    for i in range(m + 1):
        label = "" if i == 0 else a[i - 1]
        parts.append(f'<text x="{x0 - cell / 2 - 4:.1f}" y="{y0 + i * cell + cell / 2 + 4:.1f}" '
                     f'fill="#4dabf7" font-size="13" text-anchor="middle">{label}</text>')

    for i in range(m + 1):
        for j in range(n + 1):
            x, y = x0 + j * cell, y0 + i * cell
            v = d[i][j]
            if (i, j) in trace:
                fill = "#06d6a0"
                tcol = "#0d1117"
            else:
                t = v / dmax
                rr = int(0x16 + t * (0x30 - 0x16))
                gg = int(0x1b + t * (0x36 - 0x1b))
                bb = int(0x22 + t * (0x60 - 0x22))
                fill = f"#{rr:02x}{gg:02x}{bb:02x}"
                tcol = "#8b949e"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="{fill}" '
                         f'stroke="#0d1117" stroke-width="1"/>')
            parts.append(f'<text x="{x + cell/2 - 1:.1f}" y="{y + cell/2 + 4:.1f}" fill="{tcol}" '
                         f'font-size="11" text-anchor="middle">{v}</text>')

    parts.append(f'<text x="{x0 + n * cell:.1f}" y="{y0 + (m+1) * cell + 20:.1f}" fill="#06d6a0" '
                 f'font-size="11" text-anchor="end">bottom-right = edit distance ({d[m][n]})</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
