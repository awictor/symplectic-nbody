"""Demo: weighted union-find -- tracking relative offsets and catching contradictions.

Feeds a stream of 'x - y = d' difference constraints (like sensor-offset calibration), showing which
are accepted, which are rejected as contradictory, and the derived offsets. Then a parity example
(same/different groups) detects an odd cycle. Draws the accepted constraint graph with edge deltas.

    python examples/weighted_dsu_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from weighted_dsu import WeightedDSU, ParityDSU, brute_process  # noqa: E402

NAMES = ["A", "B", "C", "D", "E"]
# constraints "X - Y = d" (e.g. sensor X reads d higher than sensor Y)
CONSTRAINTS = [
    (0, 1, 3),     # A - B = 3
    (1, 2, -5),    # B - C = -5
    (0, 2, -2),    # A - C = -2  (consistent: 3 + -5 = -2)
    (2, 3, 4),     # C - D = 4
    (0, 3, 10),    # A - D = ?   A-D should be -2+4=2, so 10 CONTRADICTS
    (3, 4, 1),     # D - E = 1
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Weighted union-find: solving 'X - Y = d' difference constraints incrementally\n")

    dsu = WeightedDSU(len(NAMES))
    accepted = []
    for (x, y, d) in CONSTRAINTS:
        ok = dsu.union(x, y, d)
        tag = "accepted" if ok else "REJECTED (contradicts prior offsets)"
        print(f"  {NAMES[x]} - {NAMES[y]} = {d:3d}  ->  {tag}")
        if ok:
            accepted.append((x, y, d))

    print("\n  derived pairwise offsets (potential difference), once connected:")
    for x in range(len(NAMES)):
        for y in range(x + 1, len(NAMES)):
            dd = dsu.diff(x, y)
            if dd is not None:
                print(f"    {NAMES[x]} - {NAMES[y]} = {dd}")

    decisions = [dsu2_ok for dsu2_ok in brute_process(len(NAMES), CONSTRAINTS)]
    mine = []
    dsu2 = WeightedDSU(len(NAMES))
    for (x, y, d) in CONSTRAINTS:
        mine.append(dsu2.union(x, y, d))
    print(f"\n  decisions match brute-force BFS reference: {mine == decisions}")

    print("\n  Parity variant (same / different group), detecting an odd cycle:")
    p = ParityDSU(3)
    for (x, y, same) in [(0, 1, False), (1, 2, False), (0, 2, False)]:
        ok = p.relate(x, y, same)
        rel = "same" if same else "different"
        print(f"    {x} and {y} are {rel}  ->  {'ok' if ok else 'CONTRADICTION (odd cycle)'}")

    print("\n  Each element carries a potential relative to its set's root; find() accumulates edge")
    print("  weights to the root and compresses. A union sets the new edge weight to satisfy the")
    print("  relation; if the two are already connected it instead CHECKS consistency -- so a stream")
    print("  of relative measurements is validated in near-constant amortised time per constraint.")

    _svg(os.path.join(outdir, "weighted_dsu.svg"), accepted)
    print(f"\n  wrote {os.path.join(outdir, 'weighted_dsu.svg')}")


def _svg(path, accepted, width=680, height=380):
    import math
    n = len(NAMES)
    cx, cy, r = width / 2, height / 2 + 10, 130
    pos = {}
    for v in range(n):
        ang = 2 * math.pi * v / n - math.pi / 2
        pos[v] = (cx + r * math.cos(ang), cy + r * math.sin(ang))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Accepted difference constraints: edge label is X - Y</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'the weighted DSU keeps these consistent and rejects any contradicting offset</text>',
    ]

    for (x, y, d) in accepted:
        x1, y1 = pos[x]
        x2, y2 = pos[y]
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="#06d6a0" stroke-width="2"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        parts.append(f'<rect x="{mx-16:.0f}" y="{my-11:.0f}" width="32" height="18" '
                     f'fill="#0d1117" stroke="#30363d"/>')
        parts.append(f'<text x="{mx:.0f}" y="{my+2:.0f}" fill="#ffd43b" font-size="11" '
                     f'text-anchor="middle">{d:+d}</text>')

    for v, (x, y) in pos.items():
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="20" fill="#4dabf7"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+5:.1f}" fill="#0d1117" font-size="15" '
                     f'text-anchor="middle" font-weight="bold">{NAMES[v]}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
