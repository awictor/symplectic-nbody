"""Demo: Misra-Gries -- frequent items of a stream in tiny memory.

Finds the heavy hitters of a skewed stream with only k-1 counters (checked against exact
counting), shows the majority-vote special case, and how the summary's fixed memory beats
exact counting as the number of distinct items grows. Draws the approximate vs exact counts and
the memory comparison.

    python examples/misra_gries_demo.py [output_dir]
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from misra_gries import (MisraGries, heavy_hitters, exact_heavy_hitters,  # noqa: E402
                         majority)


def _skewed_stream(n, seed=1):
    """A Zipf-ish stream: a few items dominate, a long tail of rare ones."""
    state = seed
    out = []
    for _ in range(n):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        r = (state >> 8) / (1 << 24)
        if r < 0.35:
            out.append("A")
        elif r < 0.60:
            out.append("B")
        elif r < 0.75:
            out.append("C")
        else:
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            out.append(f"tail{(state >> 16) % 500}")  # long tail of rare items
    return out


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    stream = _skewed_stream(5000, seed=7)
    k = 4
    print(f"Misra-Gries: heavy hitters (> n/k) of a {len(stream)}-item stream with {k-1} counters\n")
    distinct = len(set(stream))
    print(f"  distinct items: {distinct} (exact counting would need {distinct} counters)")
    print(f"  Misra-Gries uses only k-1 = {k-1}\n")

    hh = heavy_hitters(stream, k)
    exact = Counter(stream)
    print(f"  {'item':>8}{'true count':>12}{'approx':>9}{'> n/k?':>9}")
    mg = MisraGries(k).update(stream)
    threshold = len(stream) / k
    for item in ("A", "B", "C"):
        approx = mg.approx_count(item)
        print(f"  {item:>8}{exact[item]:>12}{approx:>9}{'yes' if exact[item] > threshold else 'no':>9}")
    print(f"\n  heavy hitters (verified): {hh}")
    print(f"  matches exact: {hh == exact_heavy_hitters(stream, k)}\n")

    votes = "A" * 260 + "B" * 140 + "C" * 100  # A is 260/500 > half
    import random
    lst = list(votes)
    random.Random(3).shuffle(lst)
    print(f"  Majority vote (Boyer-Moore, k=2, one counter): winner = {majority(lst)} "
          f"(A has {lst.count('A')}/{len(lst)})")
    print("\n  One pass, k-1 counters, no matter how many distinct items stream by: the real")
    print("  heavy hitters are never missed (a cheap second pass drops the false positives).")
    print("  Used for network traffic monitors, trending queries, and word-frequency counting.")

    _svg(os.path.join(outdir, "misra_gries.svg"), stream, k)
    print(f"\n  wrote {os.path.join(outdir, 'misra_gries.svg')}")


def _svg(path, stream, k, w=760, h=400):
    exact = Counter(stream)
    mg = MisraGries(k).update(stream)
    n = len(stream)
    threshold = n / k

    # top items by true count
    top = [item for item, _ in exact.most_common(3)]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Misra-Gries: approximate counts track the true heavy hitters</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'true count (blue) vs the summary\'s underestimate (green); '
        f'dashed line = the n/k threshold</text>',
    ]

    # left: bar chart of true vs approx for the top items + a couple of tail items
    show = top + [it for it in list(exact)[:40] if it not in top][:3]
    lx0, lx1 = 60, w // 2 + 60
    ly0, ly1 = h - 60, 70
    vmax = max(exact[it] for it in show) * 1.1

    def LY(v):
        return ly0 - v / vmax * (ly0 - ly1)

    slot = (lx1 - lx0) / len(show)
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    # threshold line
    parts.append(f'<line x1="{lx0}" y1="{LY(threshold):.1f}" x2="{lx1}" y2="{LY(threshold):.1f}" '
                 f'stroke="#ff6b6b" stroke-width="1.2" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{lx1-2:.1f}" y="{LY(threshold)-4:.1f}" fill="#ff6b6b" font-size="9" '
                 f'text-anchor="end">n/k = {threshold:.0f}</text>')
    for i, it in enumerate(show):
        cx = lx0 + (i + 0.5) * slot
        bw = slot * 0.32
        te = exact[it]
        ap = mg.approx_count(it)
        parts.append(f'<rect x="{cx-bw-1:.1f}" y="{LY(te):.1f}" width="{bw:.1f}" '
                     f'height="{ly0-LY(te):.1f}" fill="#4dabf7"/>')
        parts.append(f'<rect x="{cx+1:.1f}" y="{LY(ap):.1f}" width="{bw:.1f}" '
                     f'height="{ly0-LY(ap):.1f}" fill="#06d6a0"/>')
        label = it if len(it) <= 4 else it[:4]
        parts.append(f'<text x="{cx:.1f}" y="{ly0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{label}</text>')
    parts.append(f'<rect x="{lx0+8}" y="{ly1}" width="9" height="9" fill="#4dabf7"/>'
                 f'<text x="{lx0+21}" y="{ly1+8}" fill="#e6edf3" font-size="9">true count</text>')
    parts.append(f'<rect x="{lx0+8}" y="{ly1+14}" width="9" height="9" fill="#06d6a0"/>'
                 f'<text x="{lx0+21}" y="{ly1+22}" fill="#e6edf3" font-size="9">MG estimate</text>')

    # right: memory comparison -- distinct items vs k-1 counters
    rx0, rx1 = w // 2 + 110, w - 30
    ry0, ry1 = h - 60, 70
    distinct = len(exact)
    bars = [("exact", distinct, "#ff922b"), (f"Misra-Gries", k - 1, "#06d6a0")]
    bmax = distinct * 1.1

    def RY(v):
        return ry0 - v / bmax * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    for i, (lab, val, col) in enumerate(bars):
        cx = rx0 + (i + 0.5) * (rx1 - rx0) / len(bars)
        bw = (rx1 - rx0) / len(bars) * 0.4
        parts.append(f'<rect x="{cx-bw/2:.1f}" y="{RY(val):.1f}" width="{bw:.1f}" '
                     f'height="{ry0-RY(val):.1f}" fill="{col}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{RY(val)-6:.1f}" fill="{col}" font-size="11" '
                     f'text-anchor="middle">{val}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{ly0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{lab}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry1-4:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">counters needed</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
