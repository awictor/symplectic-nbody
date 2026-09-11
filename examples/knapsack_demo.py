"""Demo: 0/1 knapsack -- packing the most value under a weight limit.

Solves a knapsack for the optimal value and the items to take (checked against brute force),
then shows how the optimum grows with capacity. Draws the DP value table as a heatmap and the
value-versus-capacity curve.

    python examples/knapsack_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knapsack import knapsack, knapsack_value, brute_knapsack  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    names = ["camera", "laptop", "whisky", "book", "gold bar", "phone"]
    weights = [2, 3, 4, 1, 5, 1]
    values = [6, 8, 7, 2, 12, 5]
    cap = 8

    val, items = knapsack(weights, values, cap)
    print(f"0/1 knapsack: pick items to maximize value within {cap} kg\n")
    print(f"  {'item':>10}{'weight':>8}{'value':>7}")
    for i, name in enumerate(names):
        mark = "  <- take" if i in items else ""
        print(f"  {name:>10}{weights[i]:>8}{values[i]:>7}{mark}")
    print(f"\n  optimal value {val} using {sum(weights[i] for i in items)} kg  "
          f"(brute force agrees: {brute_knapsack(weights, values, cap)[0] == val})")

    print("\n  Optimal value as the bag grows:")
    print(f"  {'capacity':>10}{'best value':>12}")
    for c in range(0, 17, 2):
        print(f"  {c:>10}{knapsack_value(weights, values, c):>12}")
    print("\n  Each cell of the DP table is 'skip item i' vs 'take it and free up its weight' --")
    print("  O(nW) instead of the 2^n subsets brute force checks. It models budget allocation,")
    print("  cargo loading, and portfolio selection under a hard cap.")

    _svg(os.path.join(outdir, "knapsack.svg"), weights, values, cap, items)
    print(f"\n  wrote {os.path.join(outdir, 'knapsack.svg')}")


def _svg(path, weights, values, cap, chosen, w=760, h=430):
    # rebuild the DP table for display
    n = len(weights)
    best = [[0] * (cap + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        wi, vi = weights[i - 1], values[i - 1]
        for cw in range(cap + 1):
            best[i][cw] = best[i - 1][cw]
            if wi <= cw and best[i - 1][cw - wi] + vi > best[i][cw]:
                best[i][cw] = best[i - 1][cw - wi] + vi
    vmax = best[n][cap] or 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'0/1 knapsack DP table: best value for i items within capacity w</text>',
        f'<text x="20" y="46" fill="#8b949e" font-size="12">'
        f'rows = items considered, columns = capacity; shade = best value; '
        f'bottom-right = the optimum</text>',
    ]

    cell = min(46, (w - 200) // (cap + 1))
    x0, y0 = 120, 80
    # column headers (capacity)
    for cw in range(cap + 1):
        parts.append(f'<text x="{x0 + cw * cell + cell / 2:.1f}" y="{y0 - 8:.1f}" fill="#8b949e" '
                     f'font-size="10" text-anchor="middle">{cw}</text>')
    parts.append(f'<text x="{x0 + (cap + 1) * cell / 2:.1f}" y="{y0 - 24:.1f}" fill="#8b949e" '
                 f'font-size="11" text-anchor="middle">capacity w</text>')
    row_labels = ["(none)"] + [f"+item{ i}" for i in range(n)]
    for i in range(n + 1):
        parts.append(f'<text x="{x0 - 8:.1f}" y="{y0 + i * cell + cell / 2 + 4:.1f}" fill="#4dabf7" '
                     f'font-size="9" text-anchor="end">{row_labels[i]}</text>')
        for cw in range(cap + 1):
            x, y = x0 + cw * cell, y0 + i * cell
            v = best[i][cw]
            t = v / vmax
            rr = int(0x16 + t * (0x8a - 0x16))
            gg = int(0x1b + t * (0xd6 - 0x1b) * 0.4)
            bb = int(0x22 + t * (0xec - 0x22))
            fill = f"#{rr:02x}{gg:02x}{bb:02x}"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="{fill}" '
                         f'stroke="#0d1117" stroke-width="1"/>')
            parts.append(f'<text x="{x + cell/2 - 1:.1f}" y="{y + cell/2 + 4:.1f}" fill="#e6edf3" '
                         f'font-size="10" text-anchor="middle">{v}</text>')
    # highlight the answer cell
    ax, ay = x0 + cap * cell, y0 + n * cell
    parts.append(f'<rect x="{ax}" y="{ay}" width="{cell-2}" height="{cell-2}" fill="none" '
                 f'stroke="#ffd43b" stroke-width="2.5"/>')

    # value vs capacity curve below
    cy0 = y0 + (n + 1) * cell + 40
    cx0, cx1 = x0, x0 + (cap + 1) * cell - cell
    caps = list(range(cap * 2 + 1))
    vals = [knapsack_value(weights, values, c) for c in caps]
    vmx = max(vals) or 1

    def CX(c):
        return cx0 + c / caps[-1] * (cx1 - cx0)

    def CY(v):
        return cy0 + 90 - v / vmx * 80

    parts.append(f'<line x1="{cx0}" y1="{cy0+90:.1f}" x2="{cx1}" y2="{cy0+90:.1f}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{cx0}" y1="{cy0+90:.1f}" x2="{cx0}" y2="{cy0:.1f}" stroke="#8b949e" stroke-width="1.2"/>')
    pts = " ".join(f"{CX(c):.1f},{CY(v):.1f}" for c, v in zip(caps, vals))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="2.5"/>')
    parts.append(f'<text x="{(cx0+cx1)/2:.1f}" y="{cy0+108:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">capacity -> optimal value (a staircase: value jumps as items become affordable)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
