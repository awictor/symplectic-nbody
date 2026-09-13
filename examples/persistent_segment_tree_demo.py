"""Demo: persistent segment tree -- range k-th smallest and full version history.

Builds a persistent count tree over an array, answers range quantile queries (k-th smallest in a
subarray) by subtracting two prefix versions, and shows that old versions of a point-add tree stay
queryable after later edits. Draws the shared-structure idea: how one update clones only a root path.

    python examples/persistent_segment_tree_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from persistent_segment_tree import PersistentSegmentTree, RangeKth  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Persistent segment tree: every update a new version, all history queryable\n")

    array = [5, 2, 8, 1, 9, 3, 7, 4, 6]
    print(f"  Array: {array}\n")
    rk = RangeKth(array)

    print("  Range k-th smallest (offline range quantile, via version_r - version_{l-1}):")
    queries = [(0, 8, 1), (0, 8, 5), (0, 8, 9), (2, 5, 2), (3, 7, 3)]
    for l, r, k in queries:
        sub = array[l:r + 1]
        print(f"    {k}-th smallest of a[{l}..{r}] = {rk.kth_smallest(l, r, k):>2}   "
              f"(subarray {sub}, sorted {sorted(sub)})")

    print("\n  Range rank (how many <= value):")
    for l, r, v in [(0, 8, 4), (2, 6, 7)]:
        print(f"    a[{l}..{r}] has {rk.range_rank(l, r, v)} elements <= {v}")

    # version history on a point-add tree
    print("\n  Version history (point-add tree; old versions immutable):")
    pst = PersistentSegmentTree(5)
    v1 = pst.update(0, 2, 10)
    v2 = pst.update(v1, 4, 7)
    v3 = pst.update(v2, 2, -3)
    print(f"    v0 (empty) total = {pst.query(0, 0, 4)}")
    print(f"    v1 (+10 @2) total = {pst.query(v1, 0, 4)}")
    print(f"    v2 (+7 @4)  total = {pst.query(v2, 0, 4)}")
    print(f"    v3 (-3 @2)  total = {pst.query(v3, 0, 4)}")
    print(f"    re-query v1 after v2,v3 exist: {pst.query(v1, 0, 4)}  (unchanged -- persistent)")
    print("\n  Each update clones only the O(log n) nodes on one root-to-leaf path; the rest is")
    print("  shared with the previous version, so n updates cost O(n log n) memory for n+1 snapshots.")

    _svg(os.path.join(outdir, "persistent_segment_tree.svg"), array, rk)
    print(f"\n  wrote {os.path.join(outdir, 'persistent_segment_tree.svg')}")


def _svg(path, array, rk, width=760, height=430):
    # visualise the k-th-smallest walk for a chosen range as a histogram of the subarray's values,
    # plus the full array with the query range highlighted.
    l, r, k = 2, 7, 3
    n = len(array)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="15">'
        f'{k}-rd smallest in a[{l}..{r}] found by differencing two prefix versions</text>',
    ]

    # top: the array as boxes, range highlighted
    bx, by, bw = 60, 70, 70
    for i, v in enumerate(array):
        inrange = l <= i <= r
        col = "#4dabf7" if inrange else "#30363d"
        x = bx + i * bw
        parts.append(f'<rect x="{x}" y="{by}" width="{bw-10}" height="44" rx="5" '
                     f'fill="#161b22" stroke="{col}" stroke-width="{2.5 if inrange else 1}"/>')
        parts.append(f'<text x="{x+(bw-10)/2:.0f}" y="{by+28:.0f}" fill="{col}" font-size="14" '
                     f'text-anchor="middle">{v}</text>')
        parts.append(f'<text x="{x+(bw-10)/2:.0f}" y="{by+58:.0f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">a{i}</text>')

    # bottom: sorted subarray with the k-th element marked
    sub = sorted(array[l:r + 1])
    ans = rk.kth_smallest(l, r, k)
    sy = 220
    parts.append(f'<text x="60" y="{sy-12}" fill="#8b949e" font-size="11">'
                 f'subarray sorted: the {k}-rd is {ans}</text>')
    sw = 70
    for i, v in enumerate(sub):
        is_ans = (i == k - 1)
        col = "#06d6a0" if is_ans else "#8b949e"
        x = 60 + i * sw
        parts.append(f'<circle cx="{x+22}" cy="{sy+25}" r="{20 if is_ans else 15}" '
                     f'fill="#161b22" stroke="{col}" stroke-width="{3 if is_ans else 1.5}"/>')
        parts.append(f'<text x="{x+22}" y="{sy+29}" fill="{col}" font-size="12" '
                     f'text-anchor="middle">{v}</text>')

    parts.append(f'<text x="60" y="{height-20}" fill="#ffd43b" font-size="12">'
                 f'answer: {ans}  (the persistent tree walks both prefix versions at once, O(log V))</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
