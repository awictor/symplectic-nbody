"""Demo: quickselect -- the k-th smallest without sorting.

Finds order statistics (min, median, percentiles, top-k) and shows quickselect touches far
fewer elements than a full sort. Draws the comparison count of quickselect vs sorting across
array sizes, and how each partition step shrinks the search range.

    python examples/quickselect_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quickselect import (quickselect, median_of_medians_select, median,  # noqa: E402
                         kth_largest, percentile, _Rng)


def _count_quickselect(data, k, seed=1):
    """Quickselect instrumented to count element comparisons."""
    a = list(data)
    n = len(a)
    rng = _Rng(seed)
    lo, hi = 0, n - 1
    comps = 0
    while lo < hi:
        pivot_index = lo + rng.randint(hi - lo + 1)
        pivot = a[pivot_index]
        a[pivot_index], a[hi] = a[hi], a[pivot_index]
        store = lo
        for i in range(lo, hi):
            comps += 1
            if a[i] < pivot:
                a[store], a[i] = a[i], a[store]
                store += 1
        a[store], a[hi] = a[hi], a[store]
        if k == store:
            break
        elif k < store:
            hi = store - 1
        else:
            lo = store + 1
    return comps


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    data = [37, 12, 91, 5, 68, 24, 50, 3, 79, 45, 18, 60]
    print("Quickselect: order statistics in O(n), no full sort\n")
    print(f"  data: {data}")
    print(f"  min           = {quickselect(data, 0)}")
    print(f"  median        = {median(data)}")
    print(f"  90th pct      = {percentile(data, 90)}")
    print(f"  3rd largest   = {kth_largest(data, 3)}")
    print(f"  (median-of-medians agrees: {median_of_medians_select(data, len(data)//2) == quickselect(data, len(data)//2)})\n")

    print("  Comparisons to find the median: quickselect (~2n) vs a full sort (~n log n):")
    print(f"  {'n':>8}{'quickselect':>14}{'sort ~n log n':>16}{'ratio':>8}")
    for n in (64, 256, 1024, 4096, 16384):
        arr = [((i * 2654435761) >> 8) % 100000 for i in range(n)]
        qc = _count_quickselect(arr, n // 2, seed=7)
        nlogn = int(n * math.log2(n))
        print(f"  {n:>8}{qc:>14}{nlogn:>16}{nlogn / qc:>7.1f}x")
    print("\n  Discarding one partition side each step gives O(n) expected time. The")
    print("  median-of-medians pivot (groups of five) guarantees it even in the worst case --")
    print("  the theoretical proof that selection beats sorting when you need just one rank.")

    _svg(os.path.join(outdir, "quickselect.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'quickselect.svg')}")


def _svg(path, w=760, h=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Quickselect: comparisons grow linearly, not n log n</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'element comparisons to find the median vs array size -- quickselect ~n, sorting ~n log n</text>',
    ]

    sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    qc, sortc = [], []
    for n in sizes:
        arr = [((i * 2654435761) >> 8) % 100000 for i in range(n)]
        qc.append(_count_quickselect(arr, n // 2, seed=7))
        sortc.append(int(n * math.log2(n)))

    lx0, lx1 = 60, w - 40
    ly0, ly1 = h - 55, 62
    ymax = max(sortc) * 1.05

    def LX(n):
        return lx0 + (math.log2(n) - math.log2(sizes[0])) / (math.log2(sizes[-1]) - math.log2(sizes[0])) * (lx1 - lx0)

    def LY(v):
        return ly0 - v / ymax * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    s = " ".join(f"{LX(n):.1f},{LY(v):.1f}" for n, v in zip(sizes, sortc))
    q = " ".join(f"{LX(n):.1f},{LY(v):.1f}" for n, v in zip(sizes, qc))
    parts.append(f'<polyline points="{s}" fill="none" stroke="#ff6b6b" stroke-width="2.5"/>')
    parts.append(f'<polyline points="{q}" fill="none" stroke="#06d6a0" stroke-width="2.5"/>')
    for n, v in zip(sizes, qc):
        parts.append(f'<circle cx="{LX(n):.1f}" cy="{LY(v):.1f}" r="2.5" fill="#06d6a0"/>')
    parts.append(f'<text x="{LX(sizes[-1]):.1f}" y="{LY(sortc[-1])-6:.1f}" fill="#ff6b6b" '
                 f'font-size="10" text-anchor="end">full sort ~n log n</text>')
    parts.append(f'<text x="{LX(sizes[-1]):.1f}" y="{LY(qc[-1])+14:.1f}" fill="#06d6a0" '
                 f'font-size="10" text-anchor="end">quickselect ~n</text>')
    for v in (0, int(ymax // 2), int(ymax)):
        parts.append(f'<text x="{lx0-6:.1f}" y="{LY(v)+3:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="end">{v}</text>')
    for n in (64, 1024, 16384):
        parts.append(f'<text x="{LX(n):.1f}" y="{ly0+15:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{n}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">array size n (log axis) -> comparisons</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
