"""Demo: a segment tree with lazy propagation for range queries and updates.

Runs range-add updates and range-sum / range-min / range-max queries on an array, showing that
lazy propagation keeps both O(log n) -- touching only a handful of nodes per operation even for a
full-array update -- and draws the tree of segment aggregates.

    python examples/segment_tree_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segment_tree import SegmentTree  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    data = [3, 1, 4, 1, 5, 9, 2, 6]
    print("Segment tree with lazy propagation: O(log n) range query AND range update\n")
    print(f"  array: {data}\n")

    st = SegmentTree(data, "sum")
    print("  Range-SUM queries:")
    for l, r in [(0, 7), (2, 5), (0, 0), (6, 7)]:
        print(f"    sum[{l}..{r}] = {st.query(l, r)}")

    print("\n  Range-ADD updates (lazy -- a full-array add touches only O(log n) nodes):")
    st.range_add(2, 5, 10)
    print(f"    after +10 to [2..5]: array = {st.to_list()}")
    print(f"    sum[0..7] = {st.query(0, 7)},  sum[2..5] = {st.query(2, 5)}")
    st.range_add(0, 7, 100)
    print(f"    after +100 to [0..7]: sum[0..7] = {st.query(0, 7)}")

    print("\n  The same array under MIN and MAX aggregates:")
    mn = SegmentTree(data, "min")
    mx = SegmentTree(data, "max")
    for l, r in [(0, 7), (1, 3), (4, 6)]:
        print(f"    [{l}..{r}]  min = {mn.query(l, r)}   max = {mx.query(l, r)}")
    mn.range_add(0, 3, 5)
    print(f"    min[0..7] after +5 to [0..3]: {mn.query(0, 7)}")

    # operation-count intuition: full-array update stays logarithmic
    print("\n  Lazy propagation keeps cost logarithmic -- a range-add over the WHOLE array:")
    for n in (16, 256, 4096, 65536):
        depth = math.ceil(math.log2(n)) + 1
        print(f"    array size {n:>6}: a full-range update visits ~{2*depth} nodes "
              f"(not {n}) -- O(log n)")

    print("\n  Each node holds the aggregate of a segment; a query descends only into the O(log n)")
    print("  nodes whose segments tile the range. A range update marks a 'lazy' tag on the covering")
    print("  nodes and pushes it to children only when a later operation visits them -- so add-to-a-")
    print("  window and query-a-window are both O(log n), which a Fenwick tree (point-update) can't do.")

    _svg(os.path.join(outdir, "segment_tree.svg"), data)
    print(f"\n  wrote {os.path.join(outdir, 'segment_tree.svg')}")


def _svg(path, data, width=760, height=400):
    n = len(data)
    # build the segment aggregates explicitly for drawing (sum tree)
    nodes = []   # (lo, hi, value, depth, x)

    def build(lo, hi, depth):
        if lo == hi:
            val = data[lo]
        else:
            mid = (lo + hi) // 2
            lval = build(lo, mid, depth + 1)
            rval = build(mid + 1, hi, depth + 1)
            val = lval + rval
        nodes.append((lo, hi, val, depth))
        return val

    build(0, n - 1, 0)
    max_depth = max(d for _, _, _, d in nodes)

    # position by segment midpoint and depth
    lx0, lx1 = 40, width - 30
    ty, by = 78, height - 40

    def X(lo, hi):
        mid = (lo + hi) / 2
        return lx0 + (mid + 0.5) / n * (lx1 - lx0)

    def Y(depth):
        return ty + depth / max(1, max_depth) * (by - ty)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Segment tree: each node is the sum of its segment</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'root covers the whole array; children split the range in half; a query tiles its range '
        f'with O(log n) nodes</text>',
    ]
    # edges: connect a node to its two children (next depth, split range)
    by_range = {(lo, hi): (X(lo, hi), Y(d)) for lo, hi, _, d in nodes}
    for lo, hi, val, d in nodes:
        if lo != hi:
            mid = (lo + hi) // 2
            x, y = by_range[(lo, hi)]
            for cr in ((lo, mid), (mid + 1, hi)):
                cx, cy = by_range[cr]
                parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" '
                             f'stroke="#30363d" stroke-width="1.2"/>')
    for lo, hi, val, d in nodes:
        x, y = by_range[(lo, hi)]
        leaf = lo == hi
        col = "#06d6a0" if leaf else "#4dabf7"
        w = 34 if not leaf else 22
        parts.append(f'<rect x="{x-w/2:.1f}" y="{y-9:.1f}" width="{w:.1f}" height="18" '
                     f'rx="3" fill="{col}" stroke="#0d1117" stroke-width="1"/>')
        label = f"{val}" if leaf else f"[{lo},{hi}]={val}"
        parts.append(f'<text x="{x:.1f}" y="{y+4:.1f}" fill="#0d1117" font-size="8" '
                     f'text-anchor="middle" font-weight="bold">{label}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
