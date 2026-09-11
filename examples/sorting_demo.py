"""Demo: comparison sorts and their trade-offs.

Measures how the four classic sorts scale in comparisons as the input grows -- the O(n log n) trio
(merge, quick, heap) pulling away from O(n^2) insertion -- shows stability (which sorts preserve the
order of equal keys), and uses the binary heap as a priority queue. Draws the comparison-count
curves on a log-log plot.

    python examples/sorting_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sorting import (insertion_sort, merge_sort, quick_sort, heap_sort,  # noqa: E402
                     BinaryHeap, Counter, ALGORITHMS, STABLE)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    state = 5

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    print("Comparison sorts: speed, stability, and worst-case trade-offs\n")
    print("  All four match Python's sorted() exactly; they differ in HOW they get there:\n")
    print(f"    {'algorithm':<10} {'time':<14} {'stable':<7} {'in-place':<9} note")
    facts = [
        ("insertion", "O(n^2)", "yes", "yes", "great on nearly-sorted / tiny arrays"),
        ("merge", "O(n log n)", "yes", "no", "O(n) scratch; never degrades"),
        ("quick", "O(n log n)*", "no", "yes", "*O(n^2) worst; median-of-3 avoids it"),
        ("heap", "O(n log n)", "no", "yes", "worst-case AND in-place"),
    ]
    for name, t, stab, ip, note in facts:
        print(f"    {name:<10} {t:<14} {stab:<7} {ip:<9} {note}")

    # scaling study
    sizes = [100, 200, 400, 800, 1600]
    curves = {name: [] for name in ALGORITHMS}
    print("\n  Comparisons vs input size (random data):")
    header = "    n     " + "".join(f"{name:>11}" for name in ALGORITHMS)
    print(header)
    for n in sizes:
        data = [int(rng() * 1000000) for _ in range(n)]
        row = f"    {n:<6}"
        for name, alg in ALGORITHMS.items():
            c = Counter()
            alg(data, counter=c)
            curves[name].append(c.comparisons)
            row += f"{c.comparisons:>11}"
        print(row)
    print("    (insertion's column grows ~4x when n doubles = O(n^2); the rest ~2x = O(n log n))")

    # stability illustration -- 40 (key, original-index) pairs with many ties
    print("\n  Stability -- sort (key, original-index) pairs by key; do equal keys keep their order?")
    tagged = [(int(rng() * 3), i) for i in range(40)]
    for name, alg in ALGORITHMS.items():
        s = alg(tagged, key=lambda p: p[0])
        violations = sum(1 for i in range(len(s) - 1)
                         if s[i][0] == s[i + 1][0] and s[i][1] > s[i + 1][1])
        tag = "stable (0 reorderings)" if violations == 0 else f"UNstable ({violations} reorderings)"
        print(f"    {name:<10}: {tag}")

    # heap as priority queue
    print("\n  The binary heap behind heap sort is also a priority queue:")
    tasks = [(3, "email"), (1, "fire"), (5, "lunch"), (2, "bug"), (4, "meeting")]
    pq = BinaryHeap(tasks, key=lambda t: t[0])
    order = []
    while len(pq):
        order.append(pq.pop()[1])
    print(f"    popped by priority: {order}")

    print("\n  Every comparison sort is bounded below by O(n log n); the interesting differences are")
    print("  the constants, the memory (merge needs scratch, heap and quick don't), stability, and")
    print("  the worst case (merge and heap never degrade; quick can without a good pivot).")

    _svg(os.path.join(outdir, "sorting.svg"), sizes, curves)
    print(f"\n  wrote {os.path.join(outdir, 'sorting.svg')}")


def _svg(path, sizes, curves, width=760, height=420):
    colors = {"insertion": "#ff6b6b", "merge": "#4dabf7", "quick": "#ffd43b", "heap": "#06d6a0"}
    lx0, lx1 = 60, width - 150
    y0, y1 = height - 55, 70
    all_counts = [c for cs in curves.values() for c in cs]
    lo = math.log10(min(all_counts))
    hi = math.log10(max(all_counts))
    lxa = math.log10(sizes[0])
    lxb = math.log10(sizes[-1])

    def X(n):
        return lx0 + (math.log10(n) - lxa) / (lxb - lxa) * (lx1 - lx0)

    def Y(c):
        return y0 - (math.log10(c) - lo) / (hi - lo) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Sort comparison counts vs input size (log-log)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'steeper slope = worse scaling: insertion O(n^2) pulls away from the O(n log n) '
        f'trio</text>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>',
    ]
    for n in sizes:
        parts.append(f'<text x="{X(n):.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{n}</text>')
    for name, cs in curves.items():
        pts = " ".join(f"{X(sizes[i]):.1f},{Y(cs[i]):.1f}" for i in range(len(sizes)))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[name]}" '
                     f'stroke-width="2.2"/>')
        for i in range(len(sizes)):
            parts.append(f'<circle cx="{X(sizes[i]):.1f}" cy="{Y(cs[i]):.1f}" r="2.6" '
                         f'fill="{colors[name]}"/>')
        parts.append(f'<text x="{lx1+8:.1f}" y="{Y(cs[-1])+3:.1f}" fill="{colors[name]}" '
                     f'font-size="11">{name}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{y0+34:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">input size n (log)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
