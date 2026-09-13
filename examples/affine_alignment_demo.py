"""Demo: affine-gap alignment (Gotoh) -- why gap opening and extension should cost differently.

Aligns two sequences differing by one long insertion, under both linear and affine gap models, showing
the affine model produces the biologically sensible single-gap alignment while linear scoring
scatters gaps. Draws the aligned sequences.

    python examples/affine_alignment_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from affine_alignment import global_align, local_align, affine_score  # noqa: E402
from sequence_alignment import needleman_wunsch  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Affine-gap alignment (Gotoh): one long gap beats many short ones\n")

    a = "ACGTACGTACGT"
    b = "ACGTACAAAGGGTACGT"  # b has an inserted block in the middle
    print(f"  Aligning:\n    a = {a}\n    b = {b}\n")

    # linear gap (NW): gap = -1 per char
    ls, la, lb = needleman_wunsch(a, b, match=2, mismatch=-1, gap=-1)
    print("  Linear gap model (each gap char costs the same):")
    print(f"    {la}\n    {lb}   score {ls}")
    print(f"    gap runs: {_gap_runs(la, lb)}")

    # affine gap: open -4, extend -1
    ags, aa, ab = global_align(a, b, match=2, mismatch=-1, gap_open=-4, gap_extend=-1)
    print("\n  Affine gap model (open -4, extend -1: one big gap is cheap):")
    print(f"    {aa}\n    {ab}   score {ags}")
    print(f"    gap runs: {_gap_runs(aa, ab)}")

    print("\n  With affine scoring the insertion aligns as ONE contiguous gap, matching the single")
    print("  biological indel event -- linear scoring has no reason to prefer that over scattered gaps.")

    # local alignment example
    x = "ZZZGATTACAZZZ"
    y = "QQGATTACAQQQQ"
    s, xa, ya = local_align(x, y, match=2, mismatch=-1, gap_open=-3, gap_extend=-1)
    print(f"\n  Local affine alignment of an embedded motif:")
    print(f"    {xa}\n    {ya}   score {s}")

    _svg(os.path.join(outdir, "affine_alignment.svg"), la, lb, aa, ab)
    print(f"\n  wrote {os.path.join(outdir, 'affine_alignment.svg')}")


def _gap_runs(aa, bb):
    runs = 0
    prev = False
    for x, y in zip(aa, bb):
        g = (x == "-" or y == "-")
        if g and not prev:
            runs += 1
        prev = g
    return runs


def _svg(path, la, lb, aa, ab, width=760, height=340):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Linear-gap alignment (top) vs affine-gap alignment (bottom, one clean block)</text>',
    ]

    def draw(a1, a2, y0, label):
        parts.append(f'<text x="30" y="{y0-8}" fill="#8b949e" font-size="11">{label}</text>')
        cw = min(26, (width - 80) // max(len(a1), 1))
        x0 = 40
        for row, seq in enumerate((a1, a2)):
            other = a2 if row == 0 else a1
            y = y0 + row * (cw + 6)
            for i, ch in enumerate(seq):
                x = x0 + i * cw
                if ch == "-":
                    col = "#ff6b6b"
                    fill = "#2a1215"
                elif i < len(other) and other[i] == ch:
                    col = "#06d6a0"
                    fill = "#0d2a20"
                else:
                    col = "#ffd43b"
                    fill = "#2a2410"
                parts.append(f'<rect x="{x}" y="{y}" width="{cw-2}" height="{cw-2}" rx="3" '
                             f'fill="{fill}" stroke="{col}" stroke-width="1"/>')
                parts.append(f'<text x="{x+(cw-2)/2:.0f}" y="{y+(cw-2)/2+4:.0f}" fill="{col}" '
                             f'font-size="11" text-anchor="middle">{ch}</text>')

    draw(la, lb, 65, "linear gap")
    draw(aa, ab, 210, "affine gap")
    parts.append('<text x="30" y="%d" fill="#8b949e" font-size="9">'
                 'green = match, yellow = mismatch, red = gap</text>' % (height - 12))
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
