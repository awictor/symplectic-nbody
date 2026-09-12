"""Demo: Hirschberg's algorithm -- optimal alignment in linear space.

Aligns two sequences optimally, verifies the score against the full Needleman-Wunsch matrix, shows the
divide-and-conquer midpoint split, and draws the alignment as a match/gap track plus the linear vs
quadratic memory saving.

    python examples/hirschberg_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hirschberg import (align, alignment_score, lcs, full_alignment, GAP,  # noqa: E402
                        _score_profile)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    a = "AGGTCACGTA"
    b = "AGCTACGCA"

    aa, bb = align(a, b)
    score = alignment_score(a, b)
    full_score, fa, fb = full_alignment(a, b)

    print("Hirschberg's algorithm: optimal alignment in linear space\n")
    print(f"  sequence A: {a}  (len {len(a)})")
    print(f"  sequence B: {b}  (len {len(b)})\n")

    print("  optimal global alignment (match +1, mismatch -1, gap -1):")
    print(f"    {aa}")
    print(f"    {''.join('|' if x == y and x != GAP else ' ' for x, y in zip(aa, bb))}")
    print(f"    {bb}")
    print(f"\n  score {score} (full Needleman-Wunsch agrees: {score == full_score})")
    print(f"  longest common subsequence: {lcs(a, b)}\n")

    # illustrate the midpoint split
    mid = len(a) // 2
    fwd = _score_profile(a[:mid], b, 1, -1, -1)
    bwd = _score_profile(a[mid:][::-1], b[::-1], 1, -1, -1)[::-1]
    best_j = max(range(len(b) + 1), key=lambda j: fwd[j] + bwd[j])
    print(f"  divide-and-conquer: split A at its midpoint (position {mid}); the optimal alignment")
    print(f"  crosses B at column {best_j}, so it recurses on A[:{mid}]xB[:{best_j}] and "
          f"A[{mid}:]xB[{best_j}:].")

    full_cells = (len(a) + 1) * (len(b) + 1)
    hb_cells = 2 * (len(b) + 1)
    print(f"\n  memory: full matrix needs {full_cells} cells; Hirschberg keeps only two rows "
          f"= {hb_cells}.")
    print("  For genome-scale inputs this is the difference between gigabytes and kilobytes, at the")
    print("  cost of only a constant factor more time -- the same optimal alignment, linear space.")

    _svg(os.path.join(outdir, "hirschberg.svg"), aa, bb)
    print(f"\n  wrote {os.path.join(outdir, 'hirschberg.svg')}")


def _svg(path, aa, bb, width=780, height=340):
    n = len(aa)
    cell = min(40, (width - 120) / n)
    ox, oy = 60, 110

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Optimal alignment: green columns match, red are mismatches, grey are gaps</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'computed in linear space by Hirschberg\'s divide-and-conquer, identical to full NW</text>',
    ]

    for i, (x, y) in enumerate(zip(aa, bb)):
        cx = ox + i * cell
        if x == GAP or y == GAP:
            col = "#8b949e"
        elif x == y:
            col = "#06d6a0"
        else:
            col = "#ff6b6b"
        # top row (A)
        parts.append(f'<rect x="{cx:.1f}" y="{oy}" width="{cell-2:.1f}" height="{cell-2:.1f}" '
                     f'fill="{col}"/>')
        parts.append(f'<text x="{cx+cell/2-1:.1f}" y="{oy+cell/2+3:.1f}" fill="#0d1117" '
                     f'font-size="14" text-anchor="middle" font-weight="bold">{x}</text>')
        # bottom row (B)
        parts.append(f'<rect x="{cx:.1f}" y="{oy+cell+6:.1f}" width="{cell-2:.1f}" '
                     f'height="{cell-2:.1f}" fill="{col}"/>')
        parts.append(f'<text x="{cx+cell/2-1:.1f}" y="{oy+cell+6+cell/2+3:.1f}" fill="#0d1117" '
                     f'font-size="14" text-anchor="middle" font-weight="bold">{y}</text>')

    parts.append(f'<text x="{ox-12}" y="{oy+cell/2+3:.0f}" fill="#8b949e" font-size="12" '
                 f'text-anchor="end">A</text>')
    parts.append(f'<text x="{ox-12}" y="{oy+cell+6+cell/2+3:.0f}" fill="#8b949e" font-size="12" '
                 f'text-anchor="end">B</text>')
    parts.append(f'<text x="{ox}" y="{oy+2*cell+40:.0f}" fill="#8b949e" font-size="12">'
                 f'linear space: only two matrix rows are ever held in memory</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
