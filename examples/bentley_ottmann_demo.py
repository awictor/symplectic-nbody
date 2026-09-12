"""Demo: sweep-line segment intersection -- finding all crossings without testing every pair.

Reports every intersecting pair of segments and their crossing points via an x-ordered plane sweep,
verifies against brute all-pairs, and draws the segments with intersection points marked.

    python examples/bentley_ottmann_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bentley_ottmann import (find_intersections, brute_find_intersections,  # noqa: E402
                             count_intersections)

SEGMENTS = [
    ((1, 1), (9, 8)),
    ((1, 8), (9, 1)),
    ((0, 4), (10, 5)),
    ((3, 0), (5, 9)),
    ((6, 0), (7, 9)),
    ((2, 6), (8, 3)),
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    inter = find_intersections(SEGMENTS)

    print("Sweep-line segment intersection: all crossings, less than all-pairs work\n")
    print(f"  {len(SEGMENTS)} segments; {len(inter)} intersecting pairs found\n")
    for a, b, pt in inter:
        loc = f"at ({pt[0]:.2f}, {pt[1]:.2f})" if pt else "(collinear overlap)"
        print(f"    segment {a} x segment {b}  {loc}")

    brute = brute_find_intersections(SEGMENTS)
    match = {(a, b) for a, b, _ in inter} == {(a, b) for a, b, _ in brute}
    print(f"\n  matches brute all-pairs test: {match}")

    # count on a big grid to show the pruning payoff
    grid = [((0, k), (20, k)) for k in range(20)] + [((k, 0), (k, 20)) for k in range(20)]
    print(f"\n  a 20x20 line grid ({len(grid)} segments): {count_intersections(grid)} crossings")
    print(f"  (all-pairs would test {len(grid)*(len(grid)-1)//2} pairs; the sweep skips every pair")
    print("  whose x-ranges don't overlap -- here the parallel lines within each family)")

    print("\n  The sweep moves a vertical line left to right, keeping only segments whose x-range")
    print("  straddles it (the active set). A new segment is tested only against those -- never")
    print("  against segments that have already ended or not yet begun -- so disjoint pairs are never")
    print("  compared, and because any crossing pair is active together at some sweep position, none")
    print("  is missed.")

    _svg(os.path.join(outdir, "bentley_ottmann.svg"), inter)
    print(f"\n  wrote {os.path.join(outdir, 'bentley_ottmann.svg')}")


def _svg(path, inter, width=460, height=460):
    lo, hi = -1, 11
    ox, oy = 30, 430
    scale = 38

    def px(x):
        return ox + (x - lo) * scale

    def py(y):
        return oy - (y - lo) * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="24" fill="#e6edf3" font-size="15">'
        f'Segment intersections found by the sweep (red dots)</text>',
        f'<text x="20" y="42" fill="#8b949e" font-size="11">'
        f'{len(inter)} crossings among the segments, each discovered exactly once</text>',
    ]

    colors = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#b197fc", "#e6edf3"]
    for k, ((x1, y1), (x2, y2)) in enumerate(SEGMENTS):
        col = colors[k % len(colors)]
        parts.append(f'<line x1="{px(x1):.1f}" y1="{py(y1):.1f}" x2="{px(x2):.1f}" '
                     f'y2="{py(y2):.1f}" stroke="{col}" stroke-width="2" opacity="0.85"/>')

    for a, b, pt in inter:
        if pt is not None:
            parts.append(f'<circle cx="{px(pt[0]):.1f}" cy="{py(pt[1]):.1f}" r="5" '
                         f'fill="#ff6b6b" stroke="#0d1117" stroke-width="1"/>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
