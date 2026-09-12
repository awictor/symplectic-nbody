"""Demo: the Li Chao tree -- the lower envelope of a bundle of lines, queried in log time.

Inserts a set of lines, queries their minimum across the x-axis, and draws the lines with their lower
envelope (the piecewise-linear convex curve the tree computes) highlighted.

    python examples/li_chao_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from li_chao import LiChaoTree, brute_min  # noqa: E402

LINES = [(-2, 40), (-1, 20), (0, 8), (1, 2), (2, 6), (3, 30)]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    xmin, xmax = -12, 12
    tree = LiChaoTree(xmin, xmax)
    for m, b in LINES:
        tree.add_line(m, b)

    print("Li Chao tree: minimum of a set of lines at any x, in O(log range)\n")
    print(f"  {len(LINES)} lines inserted (in arbitrary order):")
    for m, b in LINES:
        sign = "+" if b >= 0 else "-"
        print(f"    y = {m:+d}x {sign} {abs(b)}")

    print(f"\n  lower-envelope minimum across x:")
    print(f"    {'x':>5} {'min y':>8}  which line")
    for x in range(xmin, xmax + 1, 3):
        q = tree.query(x)
        # identify the winning line
        winner = min(LINES, key=lambda mb: mb[0] * x + mb[1])
        print(f"    {x:>5} {q:>8.1f}  y={winner[0]:+d}x{winner[1]:+d}")

    # verify against brute
    ok = all(abs(tree.query(x) - brute_min(LINES, x)) < 1e-9
             for x in range(xmin, xmax + 1))
    print(f"\n  matches brute-force min over all lines at every integer x: {ok}")
    print("\n  Each line is inserted by comparing it to the node's line at the interval midpoint,")
    print("  keeping the lower one and pushing the other into the half where it might still win --")
    print("  O(log range) per insert and per query, and unlike the monotonic-stack convex hull trick,")
    print("  it handles lines arriving in ANY order with queries interleaved. This is the standard")
    print("  speed-up that turns an O(n^2) DP with linear transition costs into O(n log n).")

    _svg(os.path.join(outdir, "li_chao.svg"), tree, xmin, xmax)
    print(f"\n  wrote {os.path.join(outdir, 'li_chao.svg')}")


def _svg(path, tree, xmin, xmax, width=760, height=440):
    # sample all lines and the envelope
    samples = 240
    xs = [xmin + (xmax - xmin) * i / samples for i in range(samples + 1)]
    env = [tree.query(x) for x in xs]

    all_y = []
    for m, b in LINES:
        for x in xs:
            all_y.append(m * x + b)
    ymin = min(min(env), min(all_y))
    ymax = max(env) + 5
    # clamp the drawing range so steep lines don't dominate
    ymax = min(ymax, max(env) + (max(env) - min(env)) * 0.6 + 10)

    ox, oy = 60, 40
    pw, ph = width - 100, height - 100

    def px(x):
        return ox + (x - xmin) / (xmax - xmin) * pw

    def pyv(y):        # small y near the bottom (SVG y grows downward, so invert)
        y = max(ymin, min(ymax, y))
        return oy + ph - (y - ymin) / (ymax - ymin) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="17">'
        f'Li Chao tree: the lower envelope (green) of a bundle of lines</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'thin blue lines are the inputs; the thick green curve is the minimum the tree returns</text>',
    ]

    # axes box
    parts.append(f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="none" '
                 f'stroke="#30363d" stroke-width="1"/>')

    # each line (thin blue)
    for m, b in LINES:
        y0 = m * xmin + b
        y1 = m * xmax + b
        parts.append(f'<line x1="{px(xmin):.1f}" y1="{pyv(y0):.1f}" x2="{px(xmax):.1f}" '
                     f'y2="{pyv(y1):.1f}" stroke="#4dabf7" stroke-width="1" opacity="0.55"/>')

    # the envelope (thick green)
    pts = " ".join(f"{px(x):.1f},{pyv(y):.1f}" for x, y in zip(xs, env))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="3"/>')

    parts.append(f'<text x="{ox+pw/2:.0f}" y="{height-20}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">x</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
