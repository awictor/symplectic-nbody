"""Demo: a splay tree and its working-set property.

Shows keys splaying to the root on access, then demonstrates the working-set advantage: with a
skewed (locality-heavy) access pattern a splay tree beats a balanced tree's fixed O(log n) depth.
Draws average access depth vs a hot-set fraction.

    python examples/splay_tree_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from splay_tree import SplayTree  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Splay tree: self-adjusting BST, hot keys rise to the root\n")

    t = SplayTree()
    for v in [50, 30, 70, 20, 40, 60, 80]:
        t.insert(v)
    print(f"  inserted 7 keys; in-order: {t.inorder()}")
    for k in [20, 80, 40]:
        _ = k in t
        print(f"    access {k} -> root is now {t.root_key()}")

    print("\n  Working-set property: with skewed access, splay beats fixed O(log n).")
    print("  Build a tree of 4000 keys, then draw accesses from a Zipf-like skewed distribution;")
    print("  measure the average depth of the accessed node BEFORE it is splayed.\n")

    n = 4000
    state = 20260911

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    def depth_of(tree, key):
        node = tree.root
        d = 0
        while node:
            if node.key == key:
                return d
            node = node.left if key < node.key else node.right
            d += 1
        return -1

    # vary how concentrated the accesses are: hot fraction of the key space
    results = []
    log2n = math.log2(n)
    for hot_frac in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
        tree = SplayTree()
        for k in range(n):
            tree.insert(k)
        hot_size = max(1, int(n * hot_frac))
        total_depth = 0
        accesses = 3000
        for _ in range(accesses):
            # 90% of accesses hit the hot set, 10% uniform
            if rng() < 0.9:
                key = int(rng() * hot_size)
            else:
                key = int(rng() * n)
            total_depth += depth_of(tree, key)
            _ = key in tree            # splay
        avg = total_depth / accesses
        results.append((hot_frac, avg))
        print(f"    hot fraction {hot_frac:5.2f} (|hot|={hot_size:4d}): "
              f"avg access depth {avg:5.2f}   (balanced tree ~ {log2n:.1f})")

    print("\n  As the access pattern concentrates, the splay tree's average access depth drops far")
    print("  below log2(n): hot keys live near the root. A balanced tree pays log2(n) every time,")
    print("  regardless of how skewed the workload is. Splaying is O(log n) amortized either way.")

    _svg(os.path.join(outdir, "splay_tree.svg"), results, log2n)
    print(f"\n  wrote {os.path.join(outdir, 'splay_tree.svg')}")


def _svg(path, results, log2n, width=760, height=430):
    m_left, m_bot, m_top, m_right = 70, 60, 80, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    # x: -log10(hot_frac) so more-skewed is to the right; y: avg depth
    xs = [-math.log10(r[0]) for r in results]
    xmin, xmax = min(xs), max(xs)
    ymax = max(max(r[1] for r in results), log2n) * 1.2

    def px(x):
        return m_left + (x - xmin) / (xmax - xmin + 1e-12) * pw

    def py(v):
        return m_top + ph - v / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Splay tree working-set property: skew lowers average access depth</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'blue = splay average access depth, yellow dashed = a balanced tree\'s fixed log2(n)</text>',
    ]

    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')

    # balanced-tree reference line
    parts.append(f'<line x1="{m_left}" y1="{py(log2n):.1f}" x2="{m_left+pw}" y2="{py(log2n):.1f}" '
                 f'stroke="#ffd43b" stroke-width="1.8" stroke-dasharray="6,4"/>')
    parts.append(f'<text x="{m_left+pw-4}" y="{py(log2n)-6:.1f}" fill="#ffd43b" font-size="11" '
                 f'text-anchor="end">balanced ~log2(n)={log2n:.1f}</text>')

    pts = " ".join(f"{px(x):.1f},{py(r[1]):.1f}" for x, r in zip(xs, results))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2.2"/>')
    for x, r in zip(xs, results):
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(r[1]):.1f}" r="4" fill="#4dabf7"/>')
        parts.append(f'<text x="{px(x):.0f}" y="{m_top+ph+18}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{r[0]:.2f}</text>')

    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height-14}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">hot-set fraction (smaller = more skewed access) -></text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
