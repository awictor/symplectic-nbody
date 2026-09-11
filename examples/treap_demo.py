"""Demo: a treap -- balance by randomization, order statistics, and split/merge.

Builds a treap, shows order statistics (select/rank), demonstrates split and merge, and empirically
shows the expected height staying near 2 log2(n) as n grows (versus the n-1 of an unbalanced BST).

    python examples/treap_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from treap import Treap, build  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Treap: a balanced BST from random priorities (tree + heap)\n")

    keys = [50, 30, 70, 20, 40, 60, 80, 10, 25, 65]
    t = build(keys, seed=3)
    print(f"  inserted {keys}")
    print(f"  in-order (sorted): {t.inorder()}")
    print(f"  height {t.height()} for {len(t)} keys\n")

    print("  Order statistics:")
    for k in [0, 4, 9]:
        print(f"    {k}-th smallest = {t.select(k)}")
    for key in [40, 65]:
        print(f"    rank of {key} = {t.rank(key)} (keys smaller than it)")

    # split / merge
    left, right = t._split(t.root, 50)
    def collect(node, out):
        if node:
            collect(node.left, out); out.append(node.key); collect(node.right, out)
    lo, hi = [], []
    collect(left, lo); collect(right, hi)
    print(f"\n  split at 50 -> lower {sorted(lo)}")
    print(f"              upper {sorted(hi)}")
    t.root = t._merge(left, right)
    print(f"  merge back  -> {t.inorder()}")

    # balance vs unbalanced insertion
    print("\n  Expected height stays near 2*log2(n) (random priorities = random insertion order):")
    print("  (an adversarial sorted insertion into a plain BST would give height n-1)")
    sizes = [100, 1000, 10000, 50000]
    heights = []
    for sz in sizes:
        tt = Treap(seed=11)
        # insert in SORTED order -- worst case for an unbalanced BST, no problem for a treap
        for i in range(sz):
            tt.insert(i)
        h = tt.height()
        heights.append((sz, h, 2 * math.log2(sz)))
        print(f"    n={sz:6d} (sorted insert): treap height {h:3d}  vs  2log2(n)={2*math.log2(sz):.1f}  "
              f"vs  unbalanced BST would be {sz-1}")

    print("\n  Every key gets a random priority; the treap is a BST on keys and a heap on priorities,")
    print("  so its shape equals a BST built from a random insertion order -- balanced with high")
    print("  probability. Split and merge (which AVL/red-black trees don't expose) fall out for free.")

    _svg(os.path.join(outdir, "treap.svg"), heights)
    print(f"\n  wrote {os.path.join(outdir, 'treap.svg')}")


def _svg(path, heights, width=760, height_px=430):
    m_left, m_bot, m_top, m_right = 70, 55, 80, 40
    pw = width - m_left - m_right
    ph = height_px - m_top - m_bot

    xs = [math.log10(h[0]) for h in heights]
    xmin, xmax = min(xs), max(xs)
    ymax = max(max(h[1] for h in heights), max(h[2] for h in heights)) * 1.2

    def px(lx):
        return m_left + (lx - xmin) / (xmax - xmin) * pw

    def py(v):
        return m_top + ph - v / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height_px}" '
        f'viewBox="0 0 {width} {height_px}" font-family="monospace">',
        f'<rect width="{width}" height="{height_px}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Treap height vs size (keys inserted in sorted order)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'blue = actual treap height, yellow dashed = 2 log2(n); a plain BST here would be height n-1</text>',
    ]

    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    for h in heights:
        x = px(math.log10(h[0]))
        parts.append(f'<text x="{x:.0f}" y="{m_top+ph+18}" fill="#8b949e" font-size="11" '
                     f'text-anchor="middle">{h[0]}</text>')
    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height_px-12}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">number of keys n (log scale)</text>')

    bpts = " ".join(f"{px(math.log10(h[0])):.1f},{py(h[2]):.1f}" for h in heights)
    parts.append(f'<polyline points="{bpts}" fill="none" stroke="#ffd43b" stroke-width="1.8" '
                 f'stroke-dasharray="6,4"/>')
    hpts = " ".join(f"{px(math.log10(h[0])):.1f},{py(h[1]):.1f}" for h in heights)
    parts.append(f'<polyline points="{hpts}" fill="none" stroke="#4dabf7" stroke-width="2.2"/>')
    for h in heights:
        parts.append(f'<circle cx="{px(math.log10(h[0])):.1f}" cy="{py(h[1]):.1f}" r="4" fill="#4dabf7"/>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
