"""Demo: longest common subsequence -- the engine behind diff.

Finds the LCS of two strings, renders a diff between two versions of a text, and shows the
insert/delete edit distance. Draws the DP length table as a heatmap with the backtrace path that
spells out the shared subsequence.

    python examples/lcs_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lcs import lcs, lcs_length, diff, apply_diff, edit_distance_indel, _table  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    a, b = "ABCBDAB", "BDCAB"
    print("Longest common subsequence: the longest in-order shared thread\n")
    print(f"  a = {a}")
    print(f"  b = {b}")
    print(f"  LCS = '{lcs(a, b)}' (length {lcs_length(a, b)}), indel edit distance "
          f"{edit_distance_indel(a, b)}\n")

    # a line-level diff, exactly how `diff`/git works
    v1 = ["import os", "def main():", "    x = 1", "    return x", "main()"]
    v2 = ["import os", "import sys", "def main():", "    x = 2", "    return x"]
    print("  A line-level diff (LCS of the two files' lines):")
    for kind, line in diff(v1, v2):
        sign = {"keep": " ", "delete": "-", "insert": "+"}[kind]
        print(f"    {sign} {line}")
    print(f"  patch reproduces v2: {apply_diff(v1, diff(v1, v2)) == v2}")
    print("\n  The kept lines are the LCS; the +/- lines are its complement -- the smallest set")
    print("  of edits. A bigger LCS means a smaller diff. This runs git, patch, and")
    print("  bioinformatics sequence comparison.")

    _svg(os.path.join(outdir, "lcs.svg"), a, b)
    print(f"\n  wrote {os.path.join(outdir, 'lcs.svg')}")


def _svg(path, a, b, w=760, h=420):
    L = _table(a, b)
    m, n = len(a), len(b)
    lmax = L[m][n] or 1

    # backtrace: the cells on the chosen LCS path
    trace = set()
    diag = set()  # cells where a match was taken
    i, j = m, n
    while i > 0 and j > 0:
        trace.add((i, j))
        if a[i - 1] == b[j - 1]:
            diag.add((i, j))
            i, j = i - 1, j - 1
        elif L[i - 1][j] >= L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    cell = 44
    x0, y0 = 100, 95
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'LCS DP table: longest shared subsequence length</text>',
        f'<text x="20" y="46" fill="#8b949e" font-size="12">'
        f'shade = LCS length of the two prefixes; yellow = match cells that build the LCS</text>',
    ]

    for j in range(n + 1):
        label = "" if j == 0 else b[j - 1]
        parts.append(f'<text x="{x0 + j * cell + cell / 2:.1f}" y="{y0 - 8:.1f}" fill="#ffd43b" '
                     f'font-size="13" text-anchor="middle">{label}</text>')
    for i in range(m + 1):
        label = "" if i == 0 else a[i - 1]
        parts.append(f'<text x="{x0 - cell / 2:.1f}" y="{y0 + i * cell + cell / 2 + 4:.1f}" '
                     f'fill="#4dabf7" font-size="13" text-anchor="middle">{label}</text>')

    for i in range(m + 1):
        for j in range(n + 1):
            x, y = x0 + j * cell, y0 + i * cell
            v = L[i][j]
            if (i, j) in diag:
                fill, tcol = "#ffd43b", "#0d1117"
            elif (i, j) in trace:
                fill, tcol = "#8338ec", "#e6edf3"
            else:
                t = v / lmax
                rr = int(0x16 + t * (0x06 - 0x16) * -1 + t * 0)  # keep it simple: blue-green ramp
                rr = int(0x16 + t * (0x2a - 0x16))
                gg = int(0x1b + t * (0x9d - 0x1b))
                bb = int(0x22 + t * (0x78 - 0x22))
                fill, tcol = f"#{rr:02x}{gg:02x}{bb:02x}", "#8b949e"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="{fill}" '
                         f'stroke="#0d1117" stroke-width="1"/>')
            parts.append(f'<text x="{x + cell/2 - 1:.1f}" y="{y + cell/2 + 4:.1f}" fill="{tcol}" '
                         f'font-size="11" text-anchor="middle">{v}</text>')

    parts.append(f'<text x="{x0 + n * cell:.1f}" y="{y0 + (m+1) * cell + 22:.1f}" fill="#ffd43b" '
                 f'font-size="11" text-anchor="end">bottom-right = LCS length ({L[m][n]}); '
                 f'yellow diagonal spells "{lcs(a, b)}"</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
