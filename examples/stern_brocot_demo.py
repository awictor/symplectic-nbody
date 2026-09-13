"""Demo: the Stern-Brocot tree approximating pi, and the Farey sequence between 0 and 1.

Descends the Stern-Brocot tree toward pi, printing the ladder of best rational approximations (22/7,
333/106, 355/113, ...) with shrinking error, shows the continued-fraction path, and lists a Farey
sequence. Draws the top levels of the Stern-Brocot tree as a mediant diagram.

    python examples/stern_brocot_demo.py [output_dir]
"""

import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stern_brocot import (  # noqa: E402
    best_rational_approximation, continued_fraction_of, farey_sequence, stern_brocot_path, mediant,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Stern-Brocot tree: every rational once, and best approximations of pi\n")

    print(f"  approximating pi = {math.pi:.10f} with growing denominator bounds:")
    print(f"    {'max denom':>10}{'fraction':>12}{'value':>14}{'error':>12}")
    seen = set()
    for N in [1, 2, 5, 8, 50, 100, 113, 300, 1000, 10000, 100000]:
        f = best_rational_approximation(math.pi, N)
        if f in seen:
            continue
        seen.add(f)
        err = abs(float(f) - math.pi)
        print(f"    {N:>10}{str(f):>12}{float(f):>14.10f}{err:>12.2e}")

    # the famous ones
    best = best_rational_approximation(math.pi, 200)
    print(f"\n  355/113 (Zu Chongzhi, 5th c.) is accurate to {abs(355/113 - math.pi):.2e}")
    print(f"  continued fraction of 355/113: {continued_fraction_of(355, 113)}")

    # Farey sequence
    print(f"\n  Farey sequence F_6 (reduced fractions in [0,1], denominator <= 6):")
    f6 = farey_sequence(6)
    print("    " + "  ".join(str(f) for f in f6))
    print(f"  consecutive pairs a/b, c/d satisfy bc - ad = 1 (unimodular) -- the tree's")
    print(f"  mediant structure restricted to a bounded denominator.")

    _svg(os.path.join(outdir, "stern_brocot.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'stern_brocot.svg')}")


def _svg(path, width=760, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Stern-Brocot tree: each node is the mediant of its two ancestors on either side</text>',
    ]

    # build the top 4 levels by mediant recursion between 0/1 and 1/0
    levels = 4
    # node = (num, den, left_bound, right_bound)
    root = (1, 1, (0, 1), (1, 0))
    nodes_by_level = [[root]]
    for _ in range(levels - 1):
        nxt = []
        for (num, den, lb, rb) in nodes_by_level[-1]:
            # left child = mediant(lb, self); right child = mediant(self, rb)
            lc = mediant(lb[0], lb[1], num, den)
            rc = mediant(num, den, rb[0], rb[1])
            nxt.append((lc[0], lc[1], lb, (num, den)))
            nxt.append((rc[0], rc[1], (num, den), rb))
        nodes_by_level.append(nxt)

    ox, oy = width / 2, 70
    dy = 75
    for lvl, nodes in enumerate(nodes_by_level):
        count = len(nodes)
        span = width - 120
        for k, (num, den, lb, rb) in enumerate(nodes):
            x = 60 + span * (k + 0.5) / count
            y = oy + lvl * dy
            # edge to parent (parent is at previous level; approximate by index//2)
            if lvl > 0:
                pcount = len(nodes_by_level[lvl - 1])
                px = 60 + span * ((k // 2) + 0.5) / pcount
                py = oy + (lvl - 1) * dy
                parts.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{x:.1f}" y2="{y:.1f}" '
                             f'stroke="#30363d" stroke-width="1"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="15" fill="#4dabf7"/>')
            parts.append(f'<text x="{x:.1f}" y="{y-1:.1f}" fill="#0d1117" font-size="10" '
                         f'text-anchor="middle">{num}</text>')
            parts.append(f'<line x1="{x-9:.1f}" y1="{y+2:.1f}" x2="{x+9:.1f}" y2="{y+2:.1f}" '
                         f'stroke="#0d1117" stroke-width="1"/>')
            parts.append(f'<text x="{x:.1f}" y="{y+11:.1f}" fill="#0d1117" font-size="10" '
                         f'text-anchor="middle">{den}</text>')
    parts.append(f'<text x="{width/2:.0f}" y="{height-16}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">left subtree = smaller values, right = larger (a BST on value)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
