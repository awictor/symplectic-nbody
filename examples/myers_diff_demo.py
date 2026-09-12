"""Demo: Myers' diff -- the shortest edit script behind git diff.

Diffs two versions of a small code file into a git-style unified listing, shows the edit distance and
LCS, confirms applying the script reproduces the new version, and draws the edit graph with the
shortest path (the diagonals are free matches).

    python examples/myers_diff_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from myers_diff import (edit_distance, edit_script, apply_script,  # noqa: E402
                        longest_common_subsequence, unified_diff, KEEP, DELETE, INSERT)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Myers' diff: the shortest edit script that turns A into B\n")

    old = [
        "def area(r):",
        "    return 3.14 * r * r",
        "",
        "def perimeter(r):",
        "    return 2 * 3.14 * r",
    ]
    new = [
        "import math",
        "",
        "def area(r):",
        "    return math.pi * r * r",
        "",
        "def perimeter(r):",
        "    return 2 * math.pi * r",
    ]

    print("  unified diff (old -> new):")
    for line in unified_diff(old, new):
        print(f"    {line}")
    d = edit_distance(old, new)
    print(f"\n  edit distance: {d} single-line edits")
    print(f"  common (kept) lines: {len(longest_common_subsequence(old, new))}")
    print(f"  applying the script reproduces new: {apply_script(old, edit_script(old, new)) == new}\n")

    # character-level diff
    a, b = "ABCABBA", "CBABAC"
    print(f"  character diff {a!r} -> {b!r}:  distance {edit_distance(a, b)}")
    steps = edit_script(a, b)
    rendered = "".join(("=" + e) if op == KEEP else ("-" + e) if op == DELETE else ("+" + e)
                       for op, e in steps)
    print(f"    script: {rendered}   (= keep, - delete, + insert)\n")

    print("  Myers models the diff as a shortest path in an edit graph: diagonal moves (matches) are")
    print("  free, right/down moves (delete/insert) cost 1. A BFS over the number of edits D finds the")
    print("  corner in O((N+M)*D) -- tiny when the files are similar, which is why git is fast.")

    _svg(os.path.join(outdir, "myers_diff.svg"), a, b)
    print(f"\n  wrote {os.path.join(outdir, 'myers_diff.svg')}")


def _svg(path, a, b, cell=42, pad=60):
    n, m = len(a), len(b)
    width = pad * 2 + m * cell
    height = pad * 2 + n * cell

    def gx(x):
        return pad + x * cell

    def gy(y):
        return pad + y * cell

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="{pad}" y="26" fill="#e6edf3" font-size="15">'
        f'Edit graph {a!r} -> {b!r}: green diagonals are free matches</text>',
    ]
    # grid
    for x in range(m + 1):
        parts.append(f'<line x1="{gx(x)}" y1="{gy(0)}" x2="{gx(x)}" y2="{gy(n)}" '
                     f'stroke="#21262d" stroke-width="1"/>')
    for y in range(n + 1):
        parts.append(f'<line x1="{gx(0)}" y1="{gy(y)}" x2="{gx(m)}" y2="{gy(y)}" '
                     f'stroke="#21262d" stroke-width="1"/>')
    # column labels (b) and row labels (a)
    for j in range(m):
        parts.append(f'<text x="{gx(j)+cell/2:.0f}" y="{pad-8}" fill="#ff922b" font-size="14" '
                     f'text-anchor="middle">{b[j]}</text>')
    for i in range(n):
        parts.append(f'<text x="{pad-14}" y="{gy(i)+cell/2+5:.0f}" fill="#4dabf7" font-size="14" '
                     f'text-anchor="middle">{a[i]}</text>')
    # matching diagonals (free edges)
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                parts.append(f'<line x1="{gx(j):.0f}" y1="{gy(i):.0f}" x2="{gx(j+1):.0f}" '
                             f'y2="{gy(i+1):.0f}" stroke="#06d6a0" stroke-width="2.5" opacity="0.4"/>')

    # the shortest path from the edit script
    script = edit_script(a, b)
    x = y = 0
    pts = [(gx(0), gy(0))]
    for op, _ in script:
        if op == KEEP:
            x += 1; y += 1
        elif op == DELETE:
            y += 1          # down = consume a
        else:
            x += 1          # right = consume b
        pts.append((gx(x), gy(y)))
    poly = " ".join(f"{px:.0f},{py:.0f}" for px, py in pts)
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ffd43b" stroke-width="3"/>')
    for px, py in pts:
        parts.append(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="3.5" fill="#ffd43b"/>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
