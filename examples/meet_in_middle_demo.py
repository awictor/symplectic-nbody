"""Demo: meet in the middle -- halving the exponent of a brute-force subset search.

Solves subset-sum, closest-sum, and knapsack-style capacity problems on values far too large for a
pseudo-polynomial DP, and shows the 2^n -> 2^(n/2) work saving. Draws the two half-sum lists combining
to hit a target.

    python examples/meet_in_middle_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from meet_in_middle import (subset_sum_exists, max_subset_sum_under, closest_subset_sum,  # noqa: E402
                            count_subsets_with_sum, _half_sums, brute_subset_sum_exists)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Meet in the middle: split the items, enumerate each half, combine\n")

    items = [7, 13, 19, 24, 31, 42, 55, 6]
    print(f"  items: {items}\n")

    for target in [50, 100, 137, 3]:
        found = subset_sum_exists(items, target)
        print(f"  subset summing to {target:3d}? {found}  (brute agrees: "
              f"{found == brute_subset_sum_exists(items, target)})")

    cap = 90
    print(f"\n  max subset sum not exceeding {cap}: {max_subset_sum_under(items, cap)}")
    print(f"  closest achievable sum to 100: {closest_subset_sum(items, 100)}")
    print(f"  subsets summing to exactly 50: {count_subsets_with_sum(items, 50)}")

    # values far too large for a pseudo-polynomial O(nW) DP
    huge = [982451653, 179424673, 512461, 15485863, 32452843, 87178291, 2038074743, 496167]
    tgt = huge[0] + huge[3] + huge[5]
    print(f"\n  values up to ~2 billion (a DP over W is hopeless):")
    print(f"    target {tgt} reachable? {subset_sum_exists(huge, tgt)}")

    n = len(items)
    print(f"\n  work saving: brute force tries 2^{n} = {2**n} subsets; meet-in-the-middle enumerates")
    print(f"  two halves of 2^{n//2} = {2**(n//2)} each = {2 * 2**(n//2)} sums, then combines by")
    print(f"  sorting + binary search. For n=40 that is a million per half, not a trillion total.")

    _svg(os.path.join(outdir, "meet_in_middle.svg"), items, 100)
    print(f"\n  wrote {os.path.join(outdir, 'meet_in_middle.svg')}")


def _svg(path, items, target, width=780, height=400):
    mid = len(items) // 2
    left = sorted(set(_half_sums(items[:mid])))
    right = sorted(set(_half_sums(items[mid:])))

    # find a left+right pair hitting the target closest
    best = None
    lset = set(left)
    for r in right:
        for cand in (target - r,):
            if cand in lset:
                best = (cand, r)
                break
        if best:
            break
    if best is None:
        # fallback: closest
        s = closest_subset_sum(items, target)
        best = (0, s)

    maxv = max(max(left), max(right), target) or 1

    def x_of(v):
        return 60 + (width - 120) * v / maxv

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Meet in the middle: left-half sums + right-half sums combine to hit {target}</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'each half is enumerated separately (2^(n/2) sums), then paired by binary search</text>',
    ]

    # target line
    tx = x_of(target)
    parts.append(f'<line x1="{tx:.1f}" y1="80" x2="{tx:.1f}" y2="320" stroke="#ffd43b" '
                 f'stroke-width="1" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{tx:.0f}" y="72" fill="#ffd43b" font-size="12" '
                 f'text-anchor="middle">target {target}</text>')

    def row(sums, y, col, label, hit):
        parts.append(f'<text x="40" y="{y+4}" fill="{col}" font-size="12" text-anchor="end">'
                     f'{label}</text>')
        for v in sums:
            x = x_of(v)
            r = 6 if v == hit else 3
            c = "#ff6b6b" if v == hit else col
            parts.append(f'<circle cx="{x:.1f}" cy="{y}" r="{r}" fill="{c}"/>')

    row(left, 150, "#4dabf7", "left sums", best[0])
    row(right, 250, "#06d6a0", "right sums", best[1])

    parts.append(f'<text x="60" y="350" fill="#8b949e" font-size="12">'
                 f'red dots: {best[0]} (left) + {best[1]} (right) = {best[0]+best[1]} matches the '
                 f'target</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
