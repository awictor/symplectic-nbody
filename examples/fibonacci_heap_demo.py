"""Demo: a Fibonacci heap, its amortized costs, and Dijkstra built on it.

Exercises insert/extract-min/decrease-key/merge, shows the maximum root degree staying within the
Fibonacci O(log n) bound as the heap grows, runs Dijkstra, and draws the root-degree-vs-size curve
against the theoretical bound.

    python examples/fibonacci_heap_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fibonacci_heap import FibonacciHeap, dijkstra  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Fibonacci heap: O(1) amortized insert, merge, and decrease-key\n")

    state = 99

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    # basic sorted extraction
    h = FibonacciHeap()
    vals = [int(rng() * 100) for _ in range(12)]
    for v in vals:
        h.insert(v)
    drained = []
    while not h.is_empty():
        drained.append(h.extract_min()[0])
    print(f"  inserted {vals}")
    print(f"  extract-min order: {drained}  (sorted: {drained == sorted(vals)})")

    # decrease-key
    h = FibonacciHeap()
    handles = {c: h.insert(k) for c, k in [("a", 50), ("b", 30), ("c", 70), ("d", 90)]}
    print(f"\n  decrease-key: min starts at {h.find_min()[0]}; ", end="")
    h.decrease_key(handles["d"], 5)
    print(f"after decreasing d(90)->5, min is {h.find_min()[0]}")

    # merge
    a = FibonacciHeap(); b = FibonacciHeap()
    for v in [4, 8, 1]:
        a.insert(v)
    for v in [3, 7, 2]:
        b.insert(v)
    a.merge(b)
    print(f"  merge two heaps -> size {len(a)}, min {a.find_min()[0]} (O(1) list concatenation)")

    # degree bound: grow the heap and record max root degree after a consolidation
    print("\n  Maximum root degree vs heap size (should stay below the Fibonacci bound log_phi(n)):")
    sizes = [10, 50, 100, 500, 1000, 5000]
    degrees = []
    phi = (1 + math.sqrt(5)) / 2
    for sz in sizes:
        hh = FibonacciHeap()
        for _ in range(sz):
            hh.insert(int(rng() * 1000000))
        hh.extract_min()      # trigger consolidation
        deg = hh.max_root_degree()
        bound = math.log(len(hh)) / math.log(phi)
        degrees.append((sz, deg, bound))
        print(f"    n={sz:5d}: max degree {deg:2d}  (bound {bound:.1f})")

    # Dijkstra
    edges = [(0, 1, 7), (0, 2, 9), (0, 5, 14), (1, 2, 10), (1, 3, 15),
             (2, 3, 11), (2, 5, 2), (3, 4, 6), (4, 5, 9)]
    undirected = edges + [(v, u, w) for u, v, w in edges]
    dist = dijkstra(6, undirected, 0)
    print(f"\n  Dijkstra shortest paths from node 0 (classic 6-node graph): {dist}")
    print(f"    (0->2->5->4 gives 9+2+9 = 20 to node 4)")

    print("\n  A Fibonacci heap stays lazy: insert and merge just splice into a circular root list,")
    print("  and decrease-key cuts a node to the roots (cascading up via mark bits). Only extract-min")
    print("  consolidates equal-degree trees -- which is what keeps decrease-key O(1) amortized and")
    print("  improves Dijkstra to O(m + n log n).")

    _svg(os.path.join(outdir, "fibonacci_heap.svg"), degrees)
    print(f"\n  wrote {os.path.join(outdir, 'fibonacci_heap.svg')}")


def _svg(path, degrees, width=760, height=430):
    m_left, m_bot, m_top, m_right = 70, 55, 80, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    xs = [math.log10(d[0]) for d in degrees]
    xmin, xmax = min(xs), max(xs)
    ymax = max(max(d[1] for d in degrees), max(d[2] for d in degrees)) * 1.15

    def px(lx):
        return m_left + (lx - xmin) / (xmax - xmin) * pw

    def py(v):
        return m_top + ph - v / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Fibonacci heap: max root degree stays within the log_phi(n) bound</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'blue = measured max root degree after consolidation, yellow dashed = the Fibonacci bound</text>',
    ]

    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    for d in degrees:
        x = px(math.log10(d[0]))
        parts.append(f'<text x="{x:.0f}" y="{m_top+ph+18}" fill="#8b949e" font-size="11" '
                     f'text-anchor="middle">{d[0]}</text>')
    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height-12}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">heap size n (log scale)</text>')

    bound_pts = " ".join(f"{px(math.log10(d[0])):.1f},{py(d[2]):.1f}" for d in degrees)
    parts.append(f'<polyline points="{bound_pts}" fill="none" stroke="#ffd43b" stroke-width="1.8" '
                 f'stroke-dasharray="6,4"/>')
    deg_pts = " ".join(f"{px(math.log10(d[0])):.1f},{py(d[1]):.1f}" for d in degrees)
    parts.append(f'<polyline points="{deg_pts}" fill="none" stroke="#4dabf7" stroke-width="2.2"/>')
    for d in degrees:
        parts.append(f'<circle cx="{px(math.log10(d[0])):.1f}" cy="{py(d[1]):.1f}" r="4" fill="#4dabf7"/>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
