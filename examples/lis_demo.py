"""Demo: longest increasing subsequence via patience sorting.

Finds the LIS of a sequence, shows the patience-sorting piles, contrasts strict vs non-decreasing and
the longest decreasing subsequence, and draws the sequence with the LIS elements highlighted.

    python examples/lis_demo.py [output_dir]
"""

import bisect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lis import lis, lis_length, longest_decreasing_subsequence  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Longest increasing subsequence: patience sorting in O(n log n)\n")

    seq = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]
    length, sub = lis(seq)
    print(f"  sequence: {seq}")
    print(f"  longest increasing subsequence (length {length}): {sub}")

    # show the patience-sorting piles
    piles = []
    for x in seq:
        i = bisect.bisect_left([p[-1] for p in piles], x)
        if i == len(piles):
            piles.append([x])
        else:
            piles[i].append(x)
    print(f"\n  patience-sorting piles (leftmost pile whose top >= card):")
    for i, p in enumerate(piles):
        print(f"    pile {i}: {p}")
    print(f"    number of piles = {len(piles)} = LIS length {length}")

    # variants
    print(f"\n  strict vs non-decreasing on [1,3,3,5,2,4,4,6]:")
    v = [1, 3, 3, 5, 2, 4, 4, 6]
    print(f"    strict:         length {lis_length(v, strict=True)}")
    print(f"    non-decreasing: length {lis_length(v, strict=False)} (equal elements allowed)")

    ld, ds = longest_decreasing_subsequence(seq)
    print(f"\n  longest DECREASING subsequence: length {ld}, {ds}")

    print("\n  Each card is placed on the leftmost pile whose top is >= it (binary search), or starts")
    print("  a new pile. The pile count equals the LIS length; back-pointers to the previous pile's")
    print("  top at placement time reconstruct the actual subsequence. O(n log n).")

    _svg(os.path.join(outdir, "lis.svg"), seq, sub)
    print(f"\n  wrote {os.path.join(outdir, 'lis.svg')}")


def _svg(path, seq, sub, width=760, height=400):
    n = len(seq)
    m_left, m_bot, m_top, m_right = 40, 70, 90, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot
    vmax = max(seq)
    bar_w = pw / n
    sub_set = set()
    # mark the actual LIS positions (match values left to right)
    idx = 0
    for x in sub:
        while idx < n and seq[idx] != x:
            idx += 1
        sub_set.add(idx)
        idx += 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Longest increasing subsequence (green bars, connected)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'each bar is a sequence element; the green ones form the longest strictly-increasing run in order</text>',
    ]
    prev_center = None
    for i, val in enumerate(seq):
        h = 10 + val / vmax * ph
        x = m_left + i * bar_w
        chosen = i in sub_set
        col = "#06d6a0" if chosen else "#30363d"
        parts.append(f'<rect x="{x + 2:.1f}" y="{m_top + ph - h:.1f}" width="{bar_w - 4:.1f}" '
                     f'height="{h:.1f}" fill="{col}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{m_top + ph + 16:.1f}" fill="#8b949e" '
                     f'font-size="11" text-anchor="middle">{val}</text>')
        if chosen:
            cx, cy = x + bar_w / 2, m_top + ph - h
            if prev_center:
                parts.append(f'<line x1="{prev_center[0]:.1f}" y1="{prev_center[1]:.1f}" '
                             f'x2="{cx:.1f}" y2="{cy:.1f}" stroke="#ffd43b" stroke-width="1.6"/>')
            prev_center = (cx, cy)

    parts.append(f'<text x="20" y="{height-16}" fill="#06d6a0" font-size="13">'
                 f'LIS length {len(sub)}: {sub}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
