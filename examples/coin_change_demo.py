"""Demo: coin change -- fewest coins and how many ways, and where greedy fails.

Computes minimum coins and the number of ways for several denomination sets, shows a case where the
greedy heuristic is beaten by DP, and contrasts combinations with ordered sequences. Draws the
min-coins-per-amount curve for two coin systems.

    python examples/coin_change_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from coin_change import min_coins, count_ways, count_sequences  # noqa: E402


def greedy_min(coins, amount):
    coins = sorted(coins, reverse=True)
    n = 0
    for c in coins:
        n += amount // c
        amount %= c
    return n if amount == 0 else None


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Coin change: fewest coins, number of ways, and the greedy trap\n")

    print("  US currency {1, 5, 10, 25} (greedy works here):")
    for amt in [63, 99, 41]:
        c, used = min_coins([1, 5, 10, 25], amt)
        print(f"    {amt}c -> {c} coins {used}")

    print("\n  The greedy trap -- coins {1, 3, 4}, make 6:")
    c, used = min_coins([1, 3, 4], 6)
    g = greedy_min([1, 3, 4], 6)
    print(f"    DP optimum: {c} coins {used}")
    print(f"    greedy (largest-first): {g} coins (4 + 1 + 1) -- WRONG, one more coin than needed")

    print("\n  Another greedy failure -- coins {1, 15, 25}, make 30:")
    c, used = min_coins([1, 15, 25], 30)
    g = greedy_min([1, 15, 25], 30)
    print(f"    DP: {c} coins {used}; greedy: {g} coins (25 + five 1s)")

    print("\n  Number of ways to make change (combinations vs ordered sequences):")
    for amt in [5, 10]:
        w = count_ways([1, 2, 5], amt)
        s = count_sequences([1, 2, 5], amt)
        print(f"    {amt} with {{1,2,5}}: {w} combinations, {s} ordered sequences")

    print("\n  The minimum-coins DP builds min[a] = 1 + min over coins c of min[a-c]. The counting DP")
    print("  puts coins in the OUTER loop (combinations) or the amount outer (sequences). Greedy is")
    print("  only correct for 'canonical' coin systems; DP is always right.")

    _svg(os.path.join(outdir, "coin_change.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'coin_change.svg')}")


def _svg(path, width=760, height=420):
    m_left, m_bot, m_top, m_right = 60, 60, 80, 130
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    amounts = list(range(1, 61))
    systems = [("US {1,5,10,25}", [1, 5, 10, 25], "#4dabf7"),
               ("{1,3,4}", [1, 3, 4], "#ffd43b"),
               ("{1,7,10}", [1, 7, 10], "#ff6b6b")]
    curves = []
    ymax = 1
    for name, coins, col in systems:
        ys = [min_coins(coins, a)[0] or 0 for a in amounts]
        ymax = max(ymax, max(ys))
        curves.append((name, col, ys))

    def px(a):
        return m_left + (a - 1) / (len(amounts) - 1) * pw

    def py(v):
        return m_top + ph - v / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Minimum coins per amount, by denomination system</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'fewer coins is better; denser systems (US) stay flat, sparse ones climb -- the DP handles all</text>',
    ]
    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    ly = m_top + 6
    for name, col, ys in curves:
        pts = " ".join(f"{px(a):.1f},{py(v):.1f}" for a, v in zip(amounts, ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.8"/>')
        parts.append(f'<line x1="{m_left+pw+12}" y1="{ly}" x2="{m_left+pw+30}" y2="{ly}" '
                     f'stroke="{col}" stroke-width="2.5"/>')
        parts.append(f'<text x="{m_left+pw+34}" y="{ly+4}" fill="#e6edf3" font-size="10">{name}</text>')
        ly += 16
    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height-12}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">amount</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
