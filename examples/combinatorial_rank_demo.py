"""Demo: combinatorial ranking -- bijecting permutations, combinations, and Gray codes to integers.

Ranks and unranks permutations and combinations, picks a huge-index permutation without enumerating,
and shows the single-bit-change Gray code. Draws the Gray-code sequence as a grid where each row
differs from the last in one cell.

    python examples/combinatorial_rank_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from combinatorial_rank import (permutation_rank, permutation_unrank, combination_rank,
                                combination_unrank, gray_encode, gray_sequence)  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Combinatorial ranking: objects <-> integers\n")

    print("  Permutations of {0,1,2,3} (lexicographic rank):")
    for r in [0, 5, 12, 23]:
        p = permutation_unrank(r, 4)
        print(f"    rank {r:2d} <-> {p}  (rank back: {permutation_rank(p)})")

    print("\n  Combinations: 3-subsets of {0..5} (combinatorial number system):")
    for r in [0, 5, 10, 19]:
        c = combination_unrank(r, 6, 3)
        print(f"    rank {r:2d} <-> {c}  (rank back: {combination_rank(c, 6)})")

    print("\n  Unranking WITHOUT enumeration -- the trillionth permutation of 15 elements:")
    idx = 1_000_000_000_000
    p = permutation_unrank(idx, 15)
    print(f"    permutation #{idx:,} of 15!: {p}")
    print(f"    (15! = {math.factorial(15):,}; we jump straight to it, no listing)")

    print("\n  Gray code -- consecutive integers whose codes differ in one bit:")
    print(f"    n:    {list(range(8))}")
    print(f"    gray: {[gray_encode(n) for n in range(8)]}")
    print(f"    binary: {[format(gray_encode(n), '03b') for n in range(8)]}")
    print("    each adjacent pair flips exactly one bit -- used in rotary encoders to avoid glitches.")

    print("\n  A rank is a unique 0..N-1 index for each object; unranking recovers it. This lets you")
    print("  store a permutation as one integer, pick a uniformly random one, or split an enumeration")
    print("  across machines by rank range -- all without materializing the (huge) full set.")

    _svg(os.path.join(outdir, "combinatorial_rank.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'combinatorial_rank.svg')}")


def _svg(path, width=760, height=430):
    bits = 5
    seq = gray_sequence(bits)
    n = len(seq)

    cell_w = 34
    cell_h = (height - 110) / n
    ox = 200
    oy = 80

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'{bits}-bit Gray code: each row flips one bit from the last</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'green cells are 1-bits; the single yellow cell each row is the bit that changed from above</text>',
    ]
    prev = None
    for i, g in enumerate(seq):
        y = oy + i * cell_h
        parts.append(f'<text x="{ox-50:.0f}" y="{y + cell_h/2 + 4:.1f}" fill="#8b949e" '
                     f'font-size="10" text-anchor="end">rank {i}</text>')
        for b in range(bits):
            bit = (g >> (bits - 1 - b)) & 1
            changed = prev is not None and ((g ^ prev) >> (bits - 1 - b)) & 1
            if changed:
                col = "#ffd43b"
            elif bit:
                col = "#06d6a0"
            else:
                col = "#161b22"
            x = ox + b * cell_w
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w-2:.1f}" '
                         f'height="{cell_h-2:.1f}" fill="{col}"/>')
        prev = g
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
