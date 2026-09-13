"""Demo: interval tree -- which intervals overlap a point or range, in O(log n + k).

Builds an interval tree over a set of "meetings", answers stabbing (who is busy at time t?) and range
(what clashes with a proposed slot?) queries, verifies against brute force, and draws the intervals
with a query highlighting the hits.

    python examples/interval_tree_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from interval_tree import IntervalTree, brute_stab, brute_overlap  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Interval tree: every interval overlapping a point or range, without scanning all\n")

    # "meetings" as (start, end, name), hours on a 24h clock
    meetings = [
        (9, 10, "standup"), (9, 12, "design"), (11, 13, "lunch-review"),
        (13, 14, "1:1"), (13, 17, "deep-work"), (15, 16, "sync"),
        (16, 18, "interview"), (20, 22, "oncall"),
    ]
    t = IntervalTree(meetings)
    print(f"  {len(t)} meetings loaded into the tree\n")

    print("  stabbing query -- who is busy at a given hour?")
    for hour in (9, 13, 15, 19):
        hits = sorted(name for _, _, name in t.stab(hour))
        assert set(n for _, _, n in t.stab(hour)) == set(n for _, _, n in brute_stab(meetings, hour))
        print(f"    at {hour}:00 -> {hits if hits else 'free'}")
    print()

    print("  range query -- what clashes with a proposed 14:00-16:00 slot?")
    clash = sorted(name for _, _, name in t.overlap(14, 16))
    assert set(n for _, _, n in t.overlap(14, 16)) == set(n for _, _, n in brute_overlap(meetings, 14, 16))
    print(f"    clashes: {clash}\n")

    print("  A centered interval tree stores intervals containing the median endpoint at each node,")
    print("  kept in two sorted lists (by start, by end). A stabbing query walks one list until it")
    print("  passes the point and recurses into only the relevant subtree -- O(log n + hits), never a")
    print("  full scan, which is how genome browsers and calendars stay fast on millions of intervals.")

    _svg(os.path.join(outdir, "interval_tree.svg"), meetings, t)
    print(f"\n  wrote {os.path.join(outdir, 'interval_tree.svg')}")


def _svg(path, meetings, tree, width=760, height=420, pad=50):
    qa, qb = 14, 16      # the range query to highlight
    hits = set(name for _, _, name in tree.overlap(qa, qb))

    lo = min(s for s, _, _ in meetings)
    hi = max(e for _, e, _ in meetings)
    span = hi - lo

    def x(t):
        return pad + (t - lo) / span * (width - 2 * pad)

    rows = sorted(meetings, key=lambda m: m[0])
    rowh = (height - 100) / len(rows)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="16">'
        f'Interval tree: meetings overlapping the 14:00-16:00 query (green)</text>',
    ]
    # query band
    parts.append(f'<rect x="{x(qa):.1f}" y="55" width="{x(qb)-x(qa):.1f}" height="{height-95}" '
                 f'fill="#06d6a0" opacity="0.12"/>')
    parts.append(f'<text x="{(x(qa)+x(qb))/2:.0f}" y="50" fill="#06d6a0" font-size="11" '
                 f'text-anchor="middle">query 14-16</text>')
    # intervals as bars
    for i, (s, e, name) in enumerate(rows):
        y = 70 + i * rowh
        col = "#06d6a0" if name in hits else "#4dabf7"
        parts.append(f'<rect x="{x(s):.1f}" y="{y:.1f}" width="{max(2,x(e)-x(s)):.1f}" '
                     f'height="{rowh*0.6:.1f}" rx="3" fill="{col}"/>')
        parts.append(f'<text x="{x(s)+3:.1f}" y="{y+rowh*0.45:.0f}" fill="#0d1117" font-size="9">'
                     f'{name}</text>')
        parts.append(f'<text x="{x(s):.1f}" y="{y-2:.0f}" fill="#8b949e" font-size="8">'
                     f'{s}-{e}</text>')
    # hour axis
    axis_y = height - 22
    for h in range(lo, hi + 1, 2):
        parts.append(f'<line x1="{x(h):.1f}" y1="{axis_y-4}" x2="{x(h):.1f}" y2="{axis_y}" '
                     f'stroke="#8b949e" stroke-width="1"/>')
        parts.append(f'<text x="{x(h):.1f}" y="{axis_y+12:.0f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{h}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
