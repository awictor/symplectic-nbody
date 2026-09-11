"""Demo: global (Needleman-Wunsch) vs local (Smith-Waterman) sequence alignment.

Aligns DNA-like and word sequences, printing the gapped alignments with a match ruler, and contrasts
global alignment (end to end) with local alignment (best-matching subsequence) on a motif buried in
mismatched flanks. Reports scores and percent identity.

    python examples/sequence_alignment_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sequence_alignment import (needleman_wunsch, smith_waterman, alignment_score,  # noqa: E402
                                identity)


def _show(aa, bb, indent="    "):
    ruler = "".join("|" if x == y and x != "-" else (" " if "-" in (x, y) else ".")
                    for x, y in zip(aa, bb))
    return f"{indent}{aa}\n{indent}{ruler}\n{indent}{bb}"


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Sequence alignment: global (Needleman-Wunsch) vs local (Smith-Waterman)\n")

    print("  Global alignment of two DNA-like sequences (aligned end to end):")
    score, aa, bb = needleman_wunsch("GCATGCU", "GATTACA")
    print(_show(aa, bb))
    print(f"    score {score}, identity {identity(aa, bb):.0%} "
          f"(| match, . mismatch, space gap)\n")

    print("  Global alignment of two words (KITTEN vs SITTING):")
    s2, a2, b2 = needleman_wunsch("KITTEN", "SITTING")
    print(_show(a2, b2))
    print(f"    score {s2}\n")

    # global vs local on an embedded motif
    x = "GGGGGATTACATTTTT"
    y = "CCGATTACAAA"
    print(f"  A shared motif GATTACA buried in different flanks:")
    print(f"    seq A = {x}")
    print(f"    seq B = {y}\n")
    gs, ga, gb = needleman_wunsch(x, y)
    ls, la, lb = smith_waterman(x, y)
    print("  GLOBAL alignment drags in the mismatched flanks:")
    print(_show(ga, gb))
    print(f"    global score {gs}\n")
    print("  LOCAL alignment isolates the conserved motif:")
    print(_show(la, lb))
    print(f"    local score {ls}, identity {identity(la, lb):.0%}\n")

    # scoring-scheme sensitivity
    print("  Gap penalty steers the alignment (AAAA vs ATAA):")
    for gp, mm in [(-1, -10), (-10, -1)]:
        s, a, b = needleman_wunsch("AAAA", "ATAA", mismatch=mm, gap=gp)
        style = "cheap gaps -> insert a gap" if gp == -1 else "costly gaps -> keep the mismatch"
        print(f"    gap={gp}, mismatch={mm}: {a} / {b}  ({style})")

    print("\n  Both fill a DP matrix where cell (i,j) is the best score aligning the first i and j")
    print("  symbols via align / gap-in-a / gap-in-b. Needleman-Wunsch seeds the borders with")
    print("  cumulative gap penalties and reads the corner (global); Smith-Waterman floors scores")
    print("  at zero and starts from the max cell (local). Traceback reconstructs the gapped strings.")

    _svg(os.path.join(outdir, "sequence_alignment.svg"), x, y, needleman_wunsch)
    print(f"\n  wrote {os.path.join(outdir, 'sequence_alignment.svg')}")


def _svg(path, a, b, aligner, width=760, height=430):
    # draw the DP score matrix as a heatmap with the traceback path highlighted
    n, m = len(a), len(b)
    match, mismatch, gap = 1, -1, -1
    H = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        H[i][0] = i * gap
    for j in range(1, m + 1):
        H[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if a[i - 1] == b[j - 1] else mismatch
            H[i][j] = max(H[i - 1][j - 1] + s, H[i - 1][j] + gap, H[i][j - 1] + gap)
    # traceback path cells
    trace = set()
    i, j = n, m
    while i > 0 or j > 0:
        trace.add((i, j))
        if i > 0 and j > 0:
            s = match if a[i - 1] == b[j - 1] else mismatch
            if H[i][j] == H[i - 1][j - 1] + s:
                i -= 1; j -= 1; continue
        if i > 0 and H[i][j] == H[i - 1][j] + gap:
            i -= 1; continue
        j -= 1
    trace.add((0, 0))

    lo = min(min(row) for row in H)
    hi = max(max(row) for row in H)
    cw = min(40, (width - 90) // (m + 1))
    ch = min(30, (height - 110) // (n + 1))
    ox, oy = 70, 80

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Needleman-Wunsch DP matrix with the traceback path</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'each cell is the best alignment score to that prefix pair; the yellow path is the '
        f'optimal alignment</text>',
    ]
    # column labels (b) and row labels (a)
    for j in range(m):
        parts.append(f'<text x="{ox + (j + 1.5) * cw:.1f}" y="{oy - 6:.1f}" fill="#8b949e" '
                     f'font-size="11" text-anchor="middle">{b[j]}</text>')
    for i in range(n):
        parts.append(f'<text x="{ox - 8:.1f}" y="{oy + (i + 1.7) * ch:.1f}" fill="#8b949e" '
                     f'font-size="11" text-anchor="end">{a[i]}</text>')
    for i in range(n + 1):
        for j in range(m + 1):
            v = H[i][j]
            t = (v - lo) / (hi - lo + 1e-9)
            if (i, j) in trace:
                fill = "#ffd43b"
            else:
                fill = f"rgb({int(30 + t * 40)},{int(50 + t * 120)},{int(60 + t * 90)})"
            x = ox + j * cw
            y = oy + i * ch
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw - 1}" height="{ch - 1}" '
                         f'fill="{fill}"/>')
            parts.append(f'<text x="{x + cw / 2:.1f}" y="{y + ch / 2 + 3:.1f}" '
                         f'fill="{"#0d1117" if (i, j) in trace else "#c9d1d9"}" font-size="8" '
                         f'text-anchor="middle">{v}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
