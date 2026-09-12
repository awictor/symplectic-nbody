"""Demo: vantage-point tree -- metric nearest-neighbour search with no coordinates.

Builds a VP-tree over 2D points and over a word list under edit distance, shows that pruned queries
match a brute-force scan while touching a fraction of the points, and draws the 2D query with its
k nearest neighbours and the shells the tree searched.

    python examples/vp_tree_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vp_tree import VPTree, euclidean, edit_distance, brute_k_nearest  # noqa: E402


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Vantage-point tree: nearest neighbours in any metric space\n")

    rng = LCG(2024)
    pts = [(rng.u() * 100, rng.u() * 100) for _ in range(1000)]
    tree = VPTree(pts, euclidean, seed=7)

    q = (50.0, 50.0)
    k = 8
    got = tree.k_nearest(q, k)
    calls = tree.distance_calls
    ref = brute_k_nearest(pts, q, k, euclidean)
    agree = all(abs(a[0] - b[0]) < 1e-9 for a, b in zip(got, ref))
    print(f"  2D Euclidean, {len(pts)} points, {k}-nearest to {q}:")
    print(f"    matches brute force: {agree}")
    print(f"    distance evaluations: {calls} of {len(pts)} points "
          f"({100*calls/len(pts):.1f}% -- the rest pruned by the triangle inequality)\n")

    # edit-distance spell-check style lookup
    words = ["algorithm", "logarithm", "rhythm", "altruism", "allegory", "analogy",
             "alignment", "argument", "arithmetic", "aligator", "allocator", "alternator",
             "gorilla", "algae", "alchemy", "almanac", "altitude", "amplitude"]
    wtree = VPTree(words, edit_distance, seed=3)
    print("  edit-distance nearest neighbours (a metric with no coordinates at all):")
    for query in ("algorith", "logarith", "alocator"):
        near = wtree.k_nearest(query, 3)
        opts = ", ".join(f"{w}({int(d)})" for d, i, w in near)
        print(f"    {query!r:14} -> {opts}")
    print()

    print("  The tree splits the space by distance to a vantage point: points nearer than the median")
    print("  go inside, farther go outside. A query prunes whole subtrees whenever the triangle")
    print("  inequality proves nothing there can beat the current best -- so it returns EXACTLY the")
    print("  brute-force answer while touching a small fraction of the data, in any metric.")

    _svg(os.path.join(outdir, "vp_tree.svg"), pts, q, got)
    print(f"\n  wrote {os.path.join(outdir, 'vp_tree.svg')}")


def _svg(path, pts, q, neighbours, size=380, pad=30):
    scale = (size - 2 * pad) / 100.0

    def sx(x):
        return pad + x * scale

    def sy(y):
        return size - pad - y * scale

    width = size + 340
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{size}" '
        f'viewBox="0 0 {width} {size}" font-family="monospace">',
        f'<rect width="{width}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="24" fill="#e6edf3" font-size="16">'
        f'VP-tree {len(neighbours)}-nearest-neighbour query</text>',
        f'<rect x="{pad}" y="{pad}" width="{size-2*pad}" height="{size-2*pad}" '
        f'fill="#010409" stroke="#30363d"/>',
    ]
    # all points faint
    for x, y in pts:
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="1.4" fill="#8b949e" opacity="0.5"/>')
    # neighbour ring: circle of the farthest neighbour distance
    far = neighbours[-1][0]
    parts.append(f'<circle cx="{sx(q[0]):.1f}" cy="{sy(q[1]):.1f}" r="{far*scale:.1f}" '
                 f'fill="none" stroke="#4dabf7" stroke-width="1" stroke-dasharray="4 3" opacity="0.7"/>')
    # neighbours highlighted
    for d, i, (x, y) in neighbours:
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="#06d6a0"/>')
    # query point
    parts.append(f'<circle cx="{sx(q[0]):.1f}" cy="{sy(q[1]):.1f}" r="5" fill="#ffd43b"/>')
    parts.append(f'<text x="{sx(q[0])+8:.0f}" y="{sy(q[1])-8:.0f}" fill="#ffd43b" font-size="11">query</text>')

    # legend / info panel
    lx = size + 15
    parts.append(f'<text x="{lx}" y="60" fill="#ffd43b" font-size="12">yellow: query point</text>')
    parts.append(f'<text x="{lx}" y="82" fill="#06d6a0" font-size="12">green: {len(neighbours)} nearest</text>')
    parts.append(f'<text x="{lx}" y="104" fill="#4dabf7" font-size="12">blue ring: k-th distance</text>')
    parts.append(f'<text x="{lx}" y="126" fill="#8b949e" font-size="12">grey: the other points</text>')
    parts.append(f'<text x="{lx}" y="164" fill="#e6edf3" font-size="12">The VP-tree indexes ANY</text>')
    parts.append(f'<text x="{lx}" y="182" fill="#e6edf3" font-size="12">metric via distances alone --</text>')
    parts.append(f'<text x="{lx}" y="200" fill="#e6edf3" font-size="12">points, strings, vectors --</text>')
    parts.append(f'<text x="{lx}" y="218" fill="#e6edf3" font-size="12">pruning by the triangle</text>')
    parts.append(f'<text x="{lx}" y="236" fill="#e6edf3" font-size="12">inequality, exact results.</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
