"""Demo: Fenwick tree -- prefix sums and point updates both in log time.

Shows the tree answering prefix and range sums, tracking point updates, and doing a cumulative
binary search. Draws the implicit tree structure (which range each node covers, by its low bit)
and the O(log n) vs O(n) cost of the two naive alternatives.

    python examples/fenwick_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fenwick import FenwickTree  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    vals = [3, 1, 4, 1, 5, 9, 2, 6]
    ft = FenwickTree.from_values(vals)
    print("Fenwick tree (binary indexed tree): update AND prefix-sum in O(log n)\n")
    print(f"  values : {vals}")
    print(f"  prefix sums : {[ft.prefix_sum(c) for c in range(1, len(vals) + 1)]}")
    print(f"  total = {ft.total()},  range_sum[2,6) = {ft.range_sum(2, 6)} "
          f"(= {sum(vals[2:6])})")

    ft.update(3, 10)
    print(f"\n  after adding 10 at index 3: value there = {ft.get(3)}, new total = {ft.total()}")

    weights = [2, 0, 3, 1, 4, 0, 5]
    fw = FenwickTree.from_values(weights)
    print(f"\n  cumulative search over weights {weights} (prefix sums "
          f"{[fw.prefix_sum(c) for c in range(1, len(weights) + 1)]}):")
    for t in (1, 3, 6, 12):
        print(f"    smallest index whose prefix sum >= {t:>2}: {fw.find_prefix(t)}")
    print("  -- the O(log n) 'select' used for weighted sampling and rank queries.\n")

    print(f"  {'n':>10}{'Fenwick log2 n':>16}{'naive prefix-array n':>22}")
    for n in (16, 1024, 1_000_000, 1_000_000_000):
        print(f"  {n:>10,}{math.ceil(math.log2(n)):>16}{n:>22,}")
    print("\n  Each update or query touches one node per set bit of the index -- log n nodes --")
    print("  in a single array of n integers. It powers range queries, order statistics, and")
    print("  streaming quantiles.")

    _svg(os.path.join(outdir, "fenwick.svg"), vals)
    print(f"\n  wrote {os.path.join(outdir, 'fenwick.svg')}")


def _svg(path, vals, w=760, h=400):
    ft = FenwickTree.from_values(vals)
    n = len(vals)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Fenwick tree: each node covers a low-bit-sized range</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'node i sums the range of length (i &amp; -i) ending at i (top); '
        f'update/query cost log n vs naive n (right)</text>',
    ]

    # top: the array cells, and above them the coverage bar for each tree node
    ax0, ay = 40, 300
    cw = 62
    for i in range(n):
        x = ax0 + i * cw
        parts.append(f'<rect x="{x}" y="{ay}" width="{cw-4}" height="34" fill="#161b22" '
                     f'stroke="#4dabf7" stroke-width="1.2"/>')
        parts.append(f'<text x="{x + (cw-4)/2:.1f}" y="{ay+22:.1f}" fill="#4dabf7" '
                     f'font-size="12" text-anchor="middle">{vals[i]}</text>')
        parts.append(f'<text x="{x + (cw-4)/2:.1f}" y="{ay+48:.1f}" fill="#8b949e" '
                     f'font-size="9" text-anchor="middle">idx {i}</text>')

    # coverage bars: internal node i (1-based) spans (i - lowbit, i]
    levels = {}  # stack bars by size for readability
    for i in range(1, n + 1):
        low = i & (-i)
        start = i - low  # 0-based start of covered range
        length = low
        level = low.bit_length()  # 1,2,4,8 -> 1,2,3,4
        y = ay - level * 34
        x = ax0 + start * cw
        bw = length * cw - 4
        col = ["#8338ec", "#b197fc", "#ff922b", "#06d6a0"][(level - 1) % 4]
        parts.append(f'<rect x="{x}" y="{y}" width="{bw:.1f}" height="26" rx="4" fill="{col}" '
                     f'opacity="0.75"/>')
        parts.append(f'<text x="{x + bw/2:.1f}" y="{y+17:.1f}" fill="#0d1117" font-size="10" '
                     f'text-anchor="middle">node {i} (len {length})</text>')

    # right-side note
    parts.append(f'<text x="{ax0}" y="{ay+72:.1f}" fill="#8b949e" font-size="10">'
                 f'A prefix sum to k adds the nodes covering [0,k): strip the low bit each step, log n adds.</text>')

    # bottom-right: cost comparison log n vs n (log-y)
    rx0, rx1 = 470, w - 30
    ry0, ry1 = 150, 70
    ns = [16, 256, 4096, 65536, 1 << 20]
    lo, hi = math.log10(ns[0]), math.log10(ns[-1])
    ymax = math.log10(ns[-1])

    def RX(nn):
        return rx0 + (math.log10(nn) - lo) / (hi - lo) * (rx1 - rx0)

    def RY(v):
        return ry0 - math.log10(max(v, 1)) / ymax * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    naive = " ".join(f"{RX(nn):.1f},{RY(nn):.1f}" for nn in ns)
    fen = " ".join(f"{RX(nn):.1f},{RY(math.log2(nn)):.1f}" for nn in ns)
    parts.append(f'<polyline points="{naive}" fill="none" stroke="#ff6b6b" stroke-width="2"/>')
    parts.append(f'<polyline points="{fen}" fill="none" stroke="#06d6a0" stroke-width="2.5"/>')
    parts.append(f'<text x="{RX(ns[-1]):.1f}" y="{RY(ns[-1])-4:.1f}" fill="#ff6b6b" font-size="9" '
                 f'text-anchor="end">naive O(n)</text>')
    parts.append(f'<text x="{RX(ns[-1]):.1f}" y="{RY(math.log2(ns[-1]))+12:.1f}" fill="#06d6a0" '
                 f'font-size="9" text-anchor="end">Fenwick O(log n)</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+16:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="middle">n (log) -> ops per query (log)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
