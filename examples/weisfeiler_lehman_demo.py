"""Demo: Weisfeiler-Lehman color refinement -- distinguishing graphs by iterated neighbourhood hashing.

Refines vertex colours on a small graph round by round, shows the partition getting finer, uses the
stable histogram to separate two non-isomorphic graphs, and shows the classic regular-graph pair that
1-WL cannot tell apart. Draws the refined colouring.

    python examples/weisfeiler_lehman_demo.py [output_dir]
"""

import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from weisfeiler_lehman import (  # noqa: E402
    refine,
    color_histogram,
    possibly_isomorphic,
    is_isomorphic_bruteforce,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Weisfeiler-Lehman color refinement: iterated neighbourhood hashing\n")

    # An asymmetric tree-ish graph where refinement splits vertices apart.
    #   0 - 1 - 2 - 3
    #       |       |
    #       4       5
    edges = [(0, 1), (1, 2), (2, 3), (1, 4), (3, 5)]
    n = 6
    print("  Graph edges:", edges)
    print("\n  Refinement (colour = hash of own colour + sorted neighbour colours):")
    prev = None
    for r in range(0, 5):
        colors, _ = refine(n, edges, max_rounds=r)
        classes = len(set(colors))
        print(f"    round {r}: colours {colors}   ({classes} class{'es' if classes != 1 else ''})")
        if colors == prev:
            print("    -> stable")
            break
        prev = colors

    # Distinguishing two graphs.
    print("\n  Distinguishing two 4-node graphs with 3 edges each:")
    path = [(0, 1), (1, 2), (2, 3)]     # P4
    star = [(0, 1), (0, 2), (0, 3)]     # K1,3
    print(f"    P4 histogram:   {color_histogram(4, path)}")
    print(f"    K1,3 histogram: {color_histogram(4, star)}")
    print(f"    WL possibly-isomorphic? {possibly_isomorphic(4, path, 4, star)}  "
          f"(exact: {is_isomorphic_bruteforce(4, path, 4, star)})")

    # The fooling case: two 2-regular graphs.
    print("\n  The 1-WL blind spot -- two 2-regular graphs:")
    c6 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    two_tri = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    print(f"    6-cycle histogram:      {color_histogram(6, c6)}")
    print(f"    two-triangles histogram:{color_histogram(6, two_tri)}")
    print(f"    WL possibly-isomorphic? {possibly_isomorphic(6, c6, 6, two_tri)}  "
          f"(exact: {is_isomorphic_bruteforce(6, c6, 6, two_tri)})")
    print("    Every vertex has degree 2 and stays one colour forever -- 1-WL is fooled, but the")
    print("    exact check knows a connected 6-cycle is not two disjoint triangles.")

    colors, _ = refine(n, edges)
    _svg(os.path.join(outdir, "weisfeiler_lehman.svg"), n, edges, colors)
    print(f"\n  wrote {os.path.join(outdir, 'weisfeiler_lehman.svg')}")


def _svg(path, n, edges, colors, width=760, height=420):
    cx, cy, R = width / 2, height / 2 + 20, 150
    pos = {i: (cx + R * math.cos(-math.pi / 2 + 2 * math.pi * i / n),
               cy + R * math.sin(-math.pi / 2 + 2 * math.pi * i / n)) for i in range(n)}
    palette = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#b197fc", "#ff6b6b", "#e6edf3", "#8b949e"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="30" fill="#e6edf3" font-size="15">'
        'Stable WL colouring (same colour = same refined neighbourhood signature)</text>',
    ]
    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                     f'stroke="#30363d" stroke-width="2"/>')
    for v in range(n):
        x, y = pos[v]
        col = palette[colors[v] % len(palette)]
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="20" fill="{col}" '
                     f'stroke="#0d1117" stroke-width="2"/>')
        parts.append(f'<text x="{x:.0f}" y="{y+4:.0f}" fill="#0d1117" font-size="12" '
                     f'text-anchor="middle" font-weight="bold">{v}</text>')
    nclasses = len(set(colors))
    parts.append(f'<text x="{width//2}" y="{height-15}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">{nclasses} colour classes after refinement</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
