"""Demo: the VCG mechanism -- efficient, truthful auctions.

Runs a second-price single-item auction and a combinatorial VCG assignment of items to bidders,
showing each winner pays its externality, then demonstrates that lying never helps a bidder. Draws the
efficient allocation and payments.

    python examples/vcg_auction_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vcg_auction import (second_price_auction, vcg_allocate, additive_value,  # noqa: E402
                         bidder_utility)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("VCG: the auction where honesty is the dominant strategy\n")

    bids = [12, 20, 15, 8]
    w, p = second_price_auction(bids)
    print(f"  single-item second-price (Vickrey) auction, bids {bids}:")
    print(f"    bidder {w} wins, pays the second-highest bid = {p}, surplus {bids[w] - p}\n")

    # combinatorial: 3 ad slots, 3 advertisers, each valuing each slot
    advertisers = ["Acme", "Bolt", "Cog"]
    slots = ["top", "mid", "low"]
    vals = [
        [30, 18, 6],    # Acme
        [24, 20, 10],   # Bolt
        [15, 14, 12],   # Cog
    ]
    r = vcg_allocate(3, [0, 1, 2], additive_value(vals))
    print(f"  combinatorial VCG: assign {slots} to {advertisers}")
    print("    value matrix (advertiser x slot):")
    print("            " + "  ".join(f"{s:>4}" for s in slots))
    for i, a in enumerate(advertisers):
        print(f"      {a:>5}  " + "   ".join(f"{vals[i][j]:>3}" for j in range(3)))
    print(f"\n    efficient allocation (maximises total value = {r['total_value']}):")
    for i, a in enumerate(advertisers):
        won = [slots[k] for k in r["allocation"][i]]
        won_str = ", ".join(won) if won else "(nothing)"
        print(f"      {a:>5}: {won_str:<12}  pays {r['payments'][i]:>4.0f}  "
              f"(externality on others), utility {r['utilities'][i]:>4.0f}")
    print()

    # strategy-proofness demonstration
    print("  can Acme do better by lying about its values?")
    truthful_util = r["utilities"][0]
    best_lie_util = truthful_util
    import itertools
    for fake in itertools.permutations([40, 25, 5]):
        lievals = [list(fake), vals[1], vals[2]]
        res = vcg_allocate(3, [0, 1, 2], additive_value(lievals))
        u = bidder_utility(res, 0, lambda b: sum(vals[0][it] for it in b))
        best_lie_util = max(best_lie_util, u)
    print(f"    truthful utility {truthful_util:.0f}; best utility over tried lies {best_lie_util:.0f}")
    print(f"    -> lying never beats the truth: {best_lie_util <= truthful_util + 1e-9}\n")

    print("  VCG picks the value-maximising allocation, then charges each winner the harm it does to")
    print("  everyone else (its externality). Because your report only changes WHETHER you win, not what")
    print("  you pay for a given outcome, truth-telling is a dominant strategy -- the principle behind")
    print("  spectrum auctions and the second-price ad exchanges that run the web.")

    _svg(os.path.join(outdir, "vcg_auction.svg"), advertisers, slots, vals, r)
    print(f"\n  wrote {os.path.join(outdir, 'vcg_auction.svg')}")


def _svg(path, advertisers, slots, vals, r, width=760, height=380):
    n = len(advertisers)
    m = len(slots)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="16">'
        f'VCG allocation: green cells are won; each winner pays its externality</text>',
    ]
    cell = 90
    ox, oy = 120, 70
    # column headers (slots)
    for j, s in enumerate(slots):
        parts.append(f'<text x="{ox + j*cell + cell/2:.0f}" y="{oy-8}" fill="#8b949e" font-size="12" '
                     f'text-anchor="middle">{s}</text>')
    won_cells = set()
    for i in range(n):
        for k in r["allocation"][i]:
            won_cells.add((i, k))
    for i, a in enumerate(advertisers):
        y = oy + i * cell
        parts.append(f'<text x="{ox-12}" y="{y+cell/2:.0f}" fill="#8b949e" font-size="12" '
                     f'text-anchor="end">{a}</text>')
        for j in range(m):
            x = ox + j * cell
            won = (i, j) in won_cells
            fill = "#06d6a0" if won else "#161b22"
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell-6:.1f}" height="{cell-6:.1f}" '
                         f'rx="4" fill="{fill}" stroke="#30363d" stroke-width="1"/>')
            tc = "#0d1117" if won else "#e6edf3"
            parts.append(f'<text x="{x+(cell-6)/2:.0f}" y="{y+(cell-6)/2+5:.0f}" fill="{tc}" '
                         f'font-size="18" text-anchor="middle">{vals[i][j]}</text>')
        # payment to the right
        px = ox + m * cell + 20
        parts.append(f'<text x="{px}" y="{y+cell/2:.0f}" fill="#ffd43b" font-size="13">'
                     f'pays {r["payments"][i]:.0f}</text>')
    parts.append(f'<text x="20" y="{height-16}" fill="#8b949e" font-size="11">'
                 f'total value maximised = {r["total_value"]:.0f}; honesty is every bidder best strategy</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
