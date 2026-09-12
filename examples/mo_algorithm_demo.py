"""Demo: Mo's algorithm -- answering a batch of range queries by reordering them.

Answers range distinct-value and power-sum queries on an array, verifies them against brute force, and
draws the window's two pointers snaking through the array in Mo's block order -- the path whose short
total length is the whole trick.

    python examples/mo_algorithm_demo.py [output_dir]
"""

import os
import sys
from math import isqrt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mo_algorithm import (range_distinct, range_power_sum,  # noqa: E402
                          brute_distinct, _mo_order)

ARR = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4]
QUERIES = [(0, 5), (3, 9), (10, 15), (0, 19), (5, 12), (14, 19), (2, 8), (7, 11)]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Mo's algorithm: many range queries in O((n+q) sqrt n) by clever ordering\n")
    print(f"  array ({len(ARR)} elements): {ARR}\n")

    distinct = range_distinct(ARR, QUERIES)
    power = range_power_sum(ARR, QUERIES)
    bd = brute_distinct(ARR, QUERIES)

    print(f"  {'range':>10}  {'distinct':>8}  {'power-sum':>9}")
    for (l, r), d, p in zip(QUERIES, distinct, power):
        print(f"  [{l:2d},{r:2d}]     {d:>8}  {p:>9}")

    print(f"\n  distinct counts match brute force: {distinct == bd}")

    # show the query processing order Mo picks
    block = max(1, isqrt(len(ARR)))
    order = _mo_order(QUERIES, block)
    print(f"\n  block size = sqrt({len(ARR)}) = {block}")
    print(f"  query order chosen (by left-block, then snaking right endpoint):")
    for oi in order:
        l, r = QUERIES[oi]
        print(f"    query {oi}: [{l},{r}]  (block {l // block})")

    # count total pointer movement in this order vs naive order
    mo_moves = _pointer_moves([QUERIES[i] for i in order])
    naive_moves = _pointer_moves(QUERIES)
    print(f"\n  total pointer travel -- Mo's order: {mo_moves}, original order: {naive_moves}")
    print("  Reordering shrinks the distance the window's two ends must travel; each step adds or")
    print("  removes one element in O(1), so total time is the total travel, O((n+q) sqrt n).")

    _svg(os.path.join(outdir, "mo_algorithm.svg"), order, block)
    print(f"\n  wrote {os.path.join(outdir, 'mo_algorithm.svg')}")


def _pointer_moves(ordered_queries):
    total = 0
    cl, cr = 0, -1
    for l, r in ordered_queries:
        total += abs(cr - r) + abs(cl - l)
        cl, cr = l, r
    return total


def _svg(path, order, block, width=780, height=440):
    n = len(ARR)
    q = len(QUERIES)
    ox, oy = 60, 80
    cell = (width - 120) / n
    row_h = (height - 140) / q

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="17">'
        f"Mo's algorithm: query windows drawn in processing order (top to bottom)</text>",
        f'<text x="20" y="49" fill="#8b949e" font-size="12">'
        f'each bar is a query range [l,r]; sorted by left-block then snaking r, so the ends move little'
        f'</text>',
    ]

    # block dividers along the array axis
    for b in range(0, n + 1, block):
        x = ox + b * cell
        parts.append(f'<line x1="{x:.1f}" y1="{oy-8}" x2="{x:.1f}" y2="{oy + q*row_h:.1f}" '
                     f'stroke="#30363d" stroke-width="1" stroke-dasharray="2 4"/>')

    # array index ticks
    for i in range(0, n, 2):
        x = ox + (i + 0.5) * cell
        parts.append(f'<text x="{x:.1f}" y="{oy-14:.0f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{i}</text>')

    palette = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#ff6b6b", "#b197fc"]
    for row, qi in enumerate(order):
        l, r = QUERIES[qi]
        y = oy + row * row_h
        x1 = ox + l * cell
        x2 = ox + (r + 1) * cell
        col = palette[(l // block) % len(palette)]
        parts.append(f'<rect x="{x1:.1f}" y="{y+3:.1f}" width="{x2-x1:.1f}" '
                     f'height="{row_h-6:.1f}" fill="{col}" opacity="0.8" rx="2"/>')
        parts.append(f'<text x="{ox - 10:.0f}" y="{y+row_h/2+3:.0f}" fill="#8b949e" '
                     f'font-size="9" text-anchor="end">q{qi}</text>')

    parts.append(f'<text x="{ox:.0f}" y="{oy + q*row_h + 28:.0f}" fill="#8b949e" font-size="11">'
                 f'dashed lines are sqrt(n)-blocks; queries in the same block are processed together'
                 f'</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
