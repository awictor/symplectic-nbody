"""Demo: cuckoo hashing -- a dictionary where EVERY lookup checks at most two cells.

Builds a cuckoo hash table, shows the two-probe guarantee, watches an insertion evict and relocate a
resident key, and compares worst-case lookup cost to a chaining table. Draws the two tables with each
key's two candidate cells.

    python examples/cuckoo_hash_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cuckoo_hash import CuckooHash  # noqa: E402


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def randint(self, lo, hi):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return lo + (self.s >> 8) % (hi - lo + 1)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Cuckoo hashing: worst-case two-probe lookups, no matter the data\n")

    ch = CuckooHash(capacity=8, seed=7)
    for k in range(12):
        ch.insert(k, k * 100)
    print(f"  inserted 12 keys into two tables of size {ch._cap} each")
    print(f"    load factor {ch.load_factor():.2f}, invariants hold = {ch.check_invariants()}\n")

    print("  every lookup checks exactly the two candidate cells h1(x) and h2(x):")
    for k in (3, 7, 99):
        i1, i2 = ch._h1(k), ch._h2(k)
        loc = ch._probe(k)
        where = "not present" if loc is None else f"table {loc[0]} cell {loc[1]}"
        print(f"    key {k:3}: candidates t1[{i1}], t2[{i2}] -> {where}")
    print()

    # worst-case comparison
    n = 100000
    print("  worst-case lookup cost vs a chaining hash table:")
    print(f"    chaining: average O(1) but a bad key can hit an O(n) chain (up to ~{n} probes)")
    print(f"    cuckoo:   ALWAYS exactly 2 cells, whatever the {n:,} keys look like\n")

    print("  Insert places x in its table-1 slot; if occupied, the resident is evicted to ITS table-2")
    print("  slot, which may evict another, ping-ponging until an empty cell is hit. A rare cycle")
    print("  triggers a rehash. Below ~50% load this settles in O(1) amortised -- so reads stay")
    print("  worst-case constant, which is what routers and hardware caches need.")

    _svg(os.path.join(outdir, "cuckoo_hash.svg"), ch)
    print(f"\n  wrote {os.path.join(outdir, 'cuckoo_hash.svg')}")


def _svg(path, ch, width=760, height=360):
    cap = ch._cap
    cell = min(70, (width - 100) / cap)
    ox = 60
    y1 = 90
    y2 = 220

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="16">'
        f'Cuckoo hash: two tables, each key lives in one of two cells</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'a lookup checks h1(x) in the top table and h2(x) in the bottom -- never more</text>',
    ]

    def draw_table(table, y, label, colour):
        out = [f'<text x="20" y="{y+cell/2+4:.0f}" fill="{colour}" font-size="12">{label}</text>']
        for i in range(cap):
            x = ox + i * cell
            filled = table[i] is not None
            fill = "#161b22"
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell-4:.1f}" height="{cell-4:.1f}" '
                       f'rx="3" fill="{fill}" stroke="{colour if filled else "#30363d"}" '
                       f'stroke-width="{2 if filled else 1}"/>')
            out.append(f'<text x="{x+(cell-4)/2:.1f}" y="{y-4:.0f}" fill="#8b949e" font-size="9" '
                       f'text-anchor="middle">{i}</text>')
            if filled:
                out.append(f'<text x="{x+(cell-4)/2:.1f}" y="{y+(cell-4)/2+4:.0f}" fill="#e6edf3" '
                           f'font-size="11" text-anchor="middle">{table[i][0]}</text>')
        return out

    parts += draw_table(ch.t1, y1, "table 1", "#4dabf7")
    parts += draw_table(ch.t2, y2, "table 2", "#06d6a0")

    # draw the two candidate cells for one example key as arrows
    k = 7
    i1, i2 = ch._h1(k), ch._h2(k)
    x1 = ox + i1 * cell + (cell - 4) / 2
    x2 = ox + i2 * cell + (cell - 4) / 2
    parts.append(f'<text x="{width-180}" y="{height-40}" fill="#ffd43b" font-size="12">'
                 f'key {k}: h1={i1}, h2={i2}</text>')
    parts.append(f'<circle cx="{x1:.1f}" cy="{y1+(cell-4)/2:.1f}" r="{(cell-4)/2+3:.1f}" '
                 f'fill="none" stroke="#ffd43b" stroke-width="2"/>')
    parts.append(f'<circle cx="{x2:.1f}" cy="{y2+(cell-4)/2:.1f}" r="{(cell-4)/2+3:.1f}" '
                 f'fill="none" stroke="#ffd43b" stroke-width="2"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
