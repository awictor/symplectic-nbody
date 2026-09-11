"""Demo: a skip list's express-lane structure and O(log n) search.

Builds a small skip list, prints its tower of shortcut lanes, traces a search hopping down from the
top level, and shows that the level distribution is geometric (half the nodes at level 1, a quarter
at level 2, ...). Draws the multi-level linked structure.

    python examples/skiplist_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from skiplist import SkipList  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    sl = SkipList(seed=5)
    keys = [3, 6, 7, 9, 12, 17, 19, 21, 25, 26]
    for k in keys:
        sl.insert(k, k)

    print("Skip list: an ordered map with express lanes for O(log n) search\n")
    print(f"  {len(sl)} keys: {sl.keys()}\n")

    # print the lanes, top to bottom
    top = sl.level
    print("  The tower of shortcut lanes (level 0 has every key; higher lanes are sparser):")
    node_lv = dict(zip(sl.keys(), sl.node_levels()))
    for lvl in range(top, -1, -1):
        lane = [k for k in sl.keys() if node_lv[k] >= lvl]
        print(f"    L{lvl}: " + "  ".join(f"{k:>2}" for k in lane))

    print("\n  Searching for 21 (drop down from the top, skip far, descend):")
    target = 21
    node = sl.head
    hops = 0
    for i in range(sl.level, -1, -1):
        while node.forward[i] is not None and node.forward[i].key < target:
            node = node.forward[i]
            hops += 1
            here = node.key
            print(f"    L{i}: advanced to {here}")
    found = node.forward[0]
    print(f"    found {target}: {found is not None and found.key == target} in {hops} forward hops "
          f"(a level-0 scan would take up to {sl.keys().index(target)})")

    # level distribution over a big list
    big = SkipList(p=0.5, seed=1)
    for k in range(4000):
        big.insert(k, k)
    levels = big.node_levels()
    print("\n  Level distribution over 4000 keys (geometric: ~half promoted each rung):")
    counts = {}
    for l in levels:
        counts[l] = counts.get(l, 0) + 1
    for l in sorted(counts):
        frac = counts[l] / len(levels)
        bar = "#" * int(round(frac * 50))
        print(f"    level {l}: {counts[l]:>4} ({frac:5.1%}) {bar}")

    print("\n  Each node is promoted to the next lane by a coin flip, so the lanes thin out")
    print("  geometrically and a search covers the list in O(log n) expected hops -- the same")
    print("  bound as a balanced tree, but with random splices instead of rotations. No")
    print("  rebalancing: insert picks a random height and links in; delete unlinks. (Redis uses it.)")

    _svg(os.path.join(outdir, "skiplist.svg"), sl, node_lv)
    print(f"\n  wrote {os.path.join(outdir, 'skiplist.svg')}")


def _svg(path, sl, node_lv, width=760, height=380):
    keys = sl.keys()
    n = len(keys)
    top = sl.level
    lx0, lx1 = 70, width - 30
    ty, by = 80, height - 60
    colw = (lx1 - lx0) / (n + 1)

    def X(idx):
        return lx0 + (idx + 1) * colw

    def Y(lvl):
        return by - lvl / max(1, top) * (by - ty)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Skip list: express lanes over a sorted linked list</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'higher lanes skip more keys; a search drops down from the top, covering the list in '
        f'O(log n) hops</text>',
    ]
    # head node column + level labels
    for lvl in range(top + 1):
        y = Y(lvl)
        parts.append(f'<text x="30" y="{y+4:.1f}" fill="#8b949e" font-size="10">L{lvl}</text>')
        parts.append(f'<rect x="{lx0-18:.1f}" y="{y-8:.1f}" width="16" height="16" '
                     f'fill="#161b22" stroke="#484f58" stroke-width="1"/>')

    # per-level forward links along the lane
    for lvl in range(top + 1):
        lane = [i for i, k in enumerate(keys) if node_lv[k] >= lvl]
        y = Y(lvl)
        prev_x = lx0 - 2
        for idx in lane:
            x = X(idx)
            parts.append(f'<line x1="{prev_x:.1f}" y1="{y:.1f}" x2="{x-9:.1f}" y2="{y:.1f}" '
                         f'stroke="#30363d" stroke-width="1.4" marker-end="url(#a)"/>')
            prev_x = x + 9

    parts.insert(4, '<defs><marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="2.5" '
                 'orient="auto"><path d="M0,0 L6,2.5 L0,5 Z" fill="#484f58"/></marker></defs>')

    # nodes: a stacked box per level it occupies
    for idx, k in enumerate(keys):
        x = X(idx)
        h = node_lv[k]
        col = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#ff6b6b", "#b197fc"][min(h, 5)]
        for lvl in range(h + 1):
            y = Y(lvl)
            parts.append(f'<rect x="{x-9:.1f}" y="{y-8:.1f}" width="18" height="16" '
                         f'fill="{col}" stroke="#0d1117" stroke-width="1" opacity="0.9"/>')
        parts.append(f'<text x="{x:.1f}" y="{Y(0)+4:.1f}" fill="#0d1117" font-size="9" '
                     f'text-anchor="middle" font-weight="bold">{k}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
