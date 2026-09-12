"""Demo: Karger's randomized min cut -- finding a network's weakest point by luck and repetition.

Runs random edge-contraction trials on a two-cluster graph, shows the best cut converging to the true
minimum as trials accumulate, checks it against exact Stoer-Wagner, and draws the graph with the cut
edges highlighted.

    python examples/karger_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from karger import karger_min_cut, karger_stein, cut_weight, _LCG, _contract_once  # noqa: E402
from stoer_wagner import min_cut as sw_min_cut  # noqa: E402

# two dense clusters {0,1,2} and {3,4,5} joined by two thin bridges
EDGES = [
    (0, 1, 4), (1, 2, 4), (2, 0, 4),          # cluster A (heavy)
    (3, 4, 4), (4, 5, 4), (5, 3, 4),          # cluster B (heavy)
    (2, 3, 1), (0, 5, 1),                      # two light bridges -> min cut 2
]
N = 6


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Karger's algorithm: minimum cut by random edge contraction\n")
    print(f"  {N} nodes in two heavy triangles joined by two light bridges (2-3 and 0-5)\n")

    exact, exact_side = sw_min_cut(N, EDGES)
    print(f"  exact minimum cut (Stoer-Wagner): {exact}, side {sorted(exact_side)}\n")

    # show convergence: best cut found after t trials
    print("  best cut found as random trials accumulate:")
    rng = _LCG(42)
    best = float("inf")
    milestones = [1, 2, 5, 10, 25, 50, 100]
    hits = 0
    for t in range(1, 101):
        cut, _ = _contract_once(N, EDGES, rng)
        if cut < best:
            best = cut
        if cut == exact:
            hits += 1
        if t in milestones:
            print(f"    after {t:3d} trials: best = {best}"
                  f"{'  <- found the true minimum' if best == exact else ''}")
    print(f"\n  {hits}/100 single trials happened to hit the minimum -- each is unlikely alone, but")
    print("  repetition makes finding it almost certain.")

    kc, (sa, sb) = karger_min_cut(N, EDGES)
    ks, _ = karger_stein(N, EDGES)
    print(f"\n  repeated Karger: {kc} (matches exact: {kc == exact})")
    print(f"  Karger-Stein:    {ks} (matches exact: {ks == exact})")
    print(f"  cut partition: {sorted(sa)} | {sorted(sb)}")

    print("\n  Each trial merges random edges until two super-nodes remain; the edges between them are")
    print("  a cut. A given minimum cut survives only if none of its few edges is ever contracted --")
    print("  probability >= 2/(n(n-1)) per run -- so O(n^2 log n) runs drive the failure chance to ~0.")

    _svg(os.path.join(outdir, "karger.svg"), sa)
    print(f"\n  wrote {os.path.join(outdir, 'karger.svg')}")


def _svg(path, side_a, width=720, height=400):
    pos = {0: (170, 90), 1: (90, 220), 2: (200, 320),
           3: (520, 320), 4: (630, 220), 5: (550, 90)}
    sa = set(side_a)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f"Karger min cut: red edges cross the cut, blue/green nodes are the two sides</text>",
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'the minimum cut severs the two light bridges, splitting the graph into its two clusters'
        f'</text>',
    ]

    for u, v, w in EDGES:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        crosses = (u in sa) != (v in sa)
        col = "#ff6b6b" if crosses else "#484f58"
        wid = 3.5 if crosses else 1.0 + w * 0.3
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
                     f'stroke-width="{wid:.1f}"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        parts.append(f'<text x="{mx:.0f}" y="{my-3:.0f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{w}</text>')

    for v, (x, y) in pos.items():
        fill = "#4dabf7" if v in sa else "#06d6a0"
        parts.append(f'<circle cx="{x}" cy="{y}" r="20" fill="{fill}"/>')
        parts.append(f'<text x="{x}" y="{y+5}" fill="#0d1117" font-size="15" '
                     f'text-anchor="middle" font-weight="bold">{v}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
