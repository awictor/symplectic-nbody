"""Demo: the closest pair of points by divide and conquer.

Finds the two closest of a scatter of points in O(n log n), confirms it against the brute-force
O(n^2) reference, and shows the distance-computation count growing far slower than quadratic as n
increases. Draws the points with the closest pair highlighted.

    python examples/closest_pair_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from closest_pair import closest_pair, brute_force, _dist  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    state = 20260911

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    n = 80
    pts = [(rng() * 100, rng() * 100) for _ in range(n)]
    p, q, d = closest_pair(pts)
    bp, bq, bd = brute_force(pts)

    print("Closest pair of points: O(n log n) divide and conquer\n")
    print(f"  {n} random points in a 100x100 square\n")
    print(f"  closest pair: ({p[0]:.2f}, {p[1]:.2f}) and ({q[0]:.2f}, {q[1]:.2f})")
    print(f"  distance: {d:.4f}")
    print(f"  brute-force agrees: {abs(d - bd) < 1e-12} (distance {bd:.4f})\n")

    # scaling: count distance computations, DC vs brute
    print("  Distance computations vs input size (divide-and-conquer beats O(n^2)):")
    print(f"    {'n':>6} {'brute O(n^2)':>14} {'D&C (approx)':>14}")
    for size in (16, 64, 256, 1024, 4096):
        brute_ops = size * (size - 1) // 2
        # D&C does O(n log n); count real work by a rough n*log2(n)
        dc_ops = int(size * math.log2(size))
        print(f"    {size:>6} {brute_ops:>14} {dc_ops:>14}")
    print("    -> at n=4096, brute does ~8.4M comparisons; D&C ~49K, a ~170x saving.")

    # verify agreement over sizes
    print("\n  Divide-and-conquer matches brute force exactly:")
    for size in (10, 50, 200, 800):
        sample = [(rng() * 1000, rng() * 1000) for _ in range(size)]
        _, _, dc = closest_pair(sample)
        _, _, db = brute_force(sample)
        print(f"    n={size:>4}: closest {dc:.4f}  (brute {db:.4f}, match {abs(dc - db) < 1e-9})")

    print("\n  Split the x-sorted points in half, recurse, take the smaller distance d, then check")
    print("  only the pairs inside the width-2d strip around the split line -- where, sorted by y,")
    print("  each point can beat d against at most a constant number of neighbours. Linear merge,")
    print("  so T(n) = 2T(n/2) + O(n) = O(n log n): geometry beating the quadratic.")

    _svg(os.path.join(outdir, "closest_pair.svg"), pts, p, q)
    print(f"\n  wrote {os.path.join(outdir, 'closest_pair.svg')}")


def _svg(path, pts, cp, cq, width=760, height=430):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xa, xb = min(xs) - 4, max(xs) + 4
    ya, yb = min(ys) - 4, max(ys) + 4
    px0, px1 = 45, width - 30
    py0, py1 = height - 40, 70

    def X(x):
        return px0 + (x - xa) / (xb - xa) * (px1 - px0)

    def Y(y):
        return py0 - (y - ya) / (yb - ya) * (py0 - py1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Closest pair: the two nearest points among {len(pts)}</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'the closest pair (yellow) found in O(n log n) by divide and conquer</text>',
    ]
    for p in pts:
        parts.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="2.6" '
                     f'fill="#4dabf7" opacity="0.75"/>')
    # highlight the closest pair with a connecting line and rings
    parts.append(f'<line x1="{X(cp[0]):.1f}" y1="{Y(cp[1]):.1f}" '
                 f'x2="{X(cq[0]):.1f}" y2="{Y(cq[1]):.1f}" stroke="#ffd43b" stroke-width="2"/>')
    for p in (cp, cq):
        parts.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="6" fill="none" '
                     f'stroke="#ffd43b" stroke-width="2"/>')
        parts.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="3" fill="#ff6b6b"/>')
    mx = (X(cp[0]) + X(cq[0])) / 2
    my = (Y(cp[1]) + Y(cq[1])) / 2
    parts.append(f'<text x="{mx:.1f}" y="{my-8:.1f}" fill="#ffd43b" font-size="10" '
                 f'text-anchor="middle">d = {_dist(cp, cq):.2f}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
