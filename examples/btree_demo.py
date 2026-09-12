"""Demo: the B-tree -- the short, fat search tree behind databases and filesystems.

Builds a B-tree, shows it stays shallow as it grows (few disk seeks for millions of keys), runs a
range query, deletes with rebalancing, and draws the multiway tree structure.

    python examples/btree_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from btree import BTree  # noqa: E402


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def randint(self, lo, hi):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return lo + (self.s >> 8) % (hi - lo + 1)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("B-tree: pack many keys per node so the tree stays short and fat\n")

    # shallowness as n grows, for a realistic disk-page fanout
    print("  height vs key count for minimum degree t=100 (a node ~ one disk page):")
    for n in (1000, 100000, 10000000, 1000000000):
        # a B-tree of degree t has height <= 1 + log_t((n+1)/2)
        t = 100
        h = 1 + math.log((n + 1) / 2, t)
        print(f"    {n:>13,} keys -> at most ~{math.ceil(h)} levels "
              f"({math.ceil(h)} disk seeks per lookup)")
    print()

    # build a small tree for the picture
    rng = LCG(2024)
    bt = BTree(t=3)
    keys = []
    while len(keys) < 24:
        k = rng.randint(1, 99)
        if k not in keys:
            keys.append(k)
            bt.insert(k, k)
    print(f"  inserted {len(keys)} keys into a t=3 tree: height {bt.height()}, "
          f"invariants hold = {bt.check_invariants()}")
    print(f"    sorted traversal: {bt.keys()}")
    print(f"    range [20, 50]:   {[k for k, _ in bt.range(20, 50)]}\n")

    # delete a batch, show rebalancing keeps it valid
    for k in keys[:10]:
        bt.delete(k)
    print(f"  after deleting 10 keys: {len(bt)} remain, height {bt.height()}, "
          f"invariants hold = {bt.check_invariants()}\n")

    print("  Every node holds up to 2t-1 keys and all leaves sit at the same depth, so the tree can")
    print("  never degenerate. Overflow splits a node and pushes its median up; underflow borrows from")
    print("  a sibling or merges. With hundreds of keys per node, a billion-row index is three seeks")
    print("  deep -- which is why B-trees index essentially every database on earth.")

    _svg(os.path.join(outdir, "btree.svg"), bt)
    print(f"\n  wrote {os.path.join(outdir, 'btree.svg')}")


def _svg(path, bt, width=760, height=340):
    # BFS layout by level
    levels = []
    frontier = [bt.root]
    while frontier:
        levels.append(frontier)
        nxt = []
        for node in frontier:
            if not node.leaf:
                nxt.extend(node.children)
        frontier = nxt

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="16">'
        f'B-tree (minimum degree t=3): each node packs several sorted keys</text>',
    ]
    pos = {}
    nlev = len(levels)
    for li, level in enumerate(levels):
        y = 70 + li * ((height - 110) / max(1, nlev - 1)) if nlev > 1 else height / 2
        # width of each node box scales with its key count
        total = sum(len(n.keys) for n in level) or 1
        x = 40
        span = width - 80
        for node in level:
            w = max(30, span * len(node.keys) / total - 12)
            label = " ".join(str(k) for k in node.keys)
            col = "#ffd43b" if node is bt.root else ("#06d6a0" if node.leaf else "#4dabf7")
            parts.append(f'<rect x="{x:.1f}" y="{y-13:.1f}" width="{w:.1f}" height="26" rx="3" '
                         f'fill="#161b22" stroke="{col}" stroke-width="2"/>')
            parts.append(f'<text x="{x+w/2:.1f}" y="{y+4:.1f}" fill="#e6edf3" font-size="10" '
                         f'text-anchor="middle">{label}</text>')
            pos[id(node)] = (x + w / 2, y, w)
            x += w + 12
    # edges
    for level in levels:
        for node in level:
            if not node.leaf:
                px, py, _ = pos[id(node)]
                for c in node.children:
                    cx, cy, _ = pos[id(c)]
                    parts.append(f'<line x1="{px:.1f}" y1="{py+13:.1f}" x2="{cx:.1f}" '
                                 f'y2="{cy-13:.1f}" stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<text x="40" y="{height-14}" fill="#8b949e" font-size="11">'
                 f'yellow: root   blue: internal   green: leaves (all at the same depth)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
