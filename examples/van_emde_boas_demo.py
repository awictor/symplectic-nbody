"""Demo: a van Emde Boas tree and its O(log log u) recursion depth.

Shows insert / successor / predecessor over an integer universe, and empirically shows the recursion
depth (operation cost) growing like log log u -- staying tiny even as the universe explodes.

    python examples/van_emde_boas_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from van_emde_boas import VEBTree  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Van Emde Boas tree: integer sets with O(log log u) successor/predecessor\n")

    v = VEBTree(64)
    for x in [5, 17, 33, 42, 9, 58, 21]:
        v.insert(x)
    print(f"  universe u=64, inserted keys, sorted: {v.to_sorted_list()}")
    print(f"  min {v.minimum()}, max {v.maximum()}")
    print(f"  successor(21) = {v.successor(21)}, successor(42) = {v.successor(42)}")
    print(f"  predecessor(33) = {v.predecessor(33)}, predecessor(5) = {v.predecessor(5)}")
    print(f"  member(17)? {17 in v}   member(18)? {18 in v}\n")

    # recursion depth vs universe size: a vEB of size u recurses on size sqrt(u)
    print("  Recursion depth per operation grows like log2(log2(u)) -- barely at all:")
    print("  (a balanced BST would pay log2(n); vEB pays log2 log2 u regardless of n)")
    depths = []
    exps = [4, 8, 16, 24, 32, 48, 64]
    for e in exps:
        u = 1 << e
        # count how many times sqrt-halving the exponent until <= 1
        depth = 0
        cur = e
        while cur > 1:
            cur = cur // 2
            depth += 1
        depths.append((e, depth, math.log2(e)))
        print(f"    u = 2^{e:2d}: vEB recursion depth {depth}   (log2 log2 u = {math.log2(e):.2f}), "
              f"BST would be up to {e} levels deep")

    # a big universe: still only a handful of steps
    big = VEBTree(1 << 24)             # ~16.7 million
    for x in [10, 5_000_000, 16_000_000, 123456, 9_999_999]:
        big.insert(x)
    print(f"\n  In a universe of 2^24 = {1<<24} values with 5 keys:")
    print(f"    successor(123456) = {big.successor(123456)}")
    print(f"    predecessor(9999999) = {big.predecessor(9999999)}")
    print(f"    sorted keys: {big.to_sorted_list()}")

    print("\n  Each key splits into a high half (which cluster) and low half (position in it); the")
    print("  tree recurses on the SQUARE ROOT of the universe, and storing min/max directly caps the")
    print("  recursion at one call per level -- so T(u) = T(sqrt u) + O(1) = O(log log u).")

    _svg(os.path.join(outdir, "van_emde_boas.svg"), depths)
    print(f"\n  wrote {os.path.join(outdir, 'van_emde_boas.svg')}")


def _svg(path, depths, width=760, height=430):
    m_left, m_bot, m_top, m_right = 70, 60, 80, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    xs = [d[0] for d in depths]
    xmin, xmax = min(xs), max(xs)
    ymax = max(max(d[0] for d in depths), max(d[1] for d in depths)) * 1.05

    def px(e):
        return m_left + (e - xmin) / (xmax - xmin) * pw

    def py(v):
        return m_top + ph - v / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Van Emde Boas: O(log log u) depth vs a BST\'s O(log u)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'blue = vEB recursion depth (log log u), red = a BST\'s depth over the same universe (log u)</text>',
    ]

    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    for d in depths:
        x = px(d[0])
        parts.append(f'<text x="{x:.0f}" y="{m_top+ph+18}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">2^{d[0]}</text>')
    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height-14}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">universe size u</text>')

    # BST depth = log2(u) = the exponent
    bst = " ".join(f"{px(d[0]):.1f},{py(d[0]):.1f}" for d in depths)
    parts.append(f'<polyline points="{bst}" fill="none" stroke="#ff6b6b" stroke-width="2" '
                 f'stroke-dasharray="6,4"/>')
    parts.append(f'<text x="{px(depths[-1][0]):.0f}" y="{py(depths[-1][0])-8:.0f}" fill="#ff6b6b" '
                 f'font-size="11" text-anchor="end">BST ~ log u</text>')

    # vEB depth
    veb = " ".join(f"{px(d[0]):.1f},{py(d[1]):.1f}" for d in depths)
    parts.append(f'<polyline points="{veb}" fill="none" stroke="#4dabf7" stroke-width="2.4"/>')
    for d in depths:
        parts.append(f'<circle cx="{px(d[0]):.1f}" cy="{py(d[1]):.1f}" r="4" fill="#4dabf7"/>')
    parts.append(f'<text x="{px(depths[-1][0]):.0f}" y="{py(depths[-1][1])+18:.0f}" fill="#4dabf7" '
                 f'font-size="11" text-anchor="end">vEB ~ log log u</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
