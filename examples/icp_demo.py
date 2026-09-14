"""Demo: Iterative Closest Point registering two point clouds with unknown correspondence.

Takes a point cloud, applies an unknown rotation + translation, shuffles the point order, and lets ICP
recover the transform from scratch -- matching nearest neighbours and re-aligning until the clouds
snap together. Shows the RMSD collapsing over iterations and draws the source, target, and aligned
clouds.

    python examples/icp_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from icp import icp, rotation_2d, _nearest_neighbours, _mean_sq_error  # noqa: E402
from kabsch import apply_transform  # noqa: E402


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Iterative Closest Point: registering clouds with no known correspondence\n")

    rng = _lcg(2024)
    # a distinctive 2-D shape (an L)
    src = [[x, 0.0] for x in range(6)] + [[0.0, y] for y in range(1, 5)]
    src = [[p[0] + 0.1 * (rng() - 0.5), p[1] + 0.1 * (rng() - 0.5)] for p in src]

    theta = 0.35
    R = rotation_2d(theta)
    t = [4.0, -2.0]
    target = apply_transform(R, t, src)
    # shuffle the target point order -- ICP does not get the correspondence
    perm = list(range(len(target)))
    for i in range(len(perm) - 1, 0, -1):
        j = int(rng() * (i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    shuffled = [target[perm[i]] for i in range(len(target))]

    print(f"  source: {len(src)}-point L-shape")
    print(f"  target: same shape rotated {theta} rad, translated {t}, point order SHUFFLED\n")

    res = icp(src, shuffled, max_iter=40)
    print(f"  ICP iterations: {res['iterations']}")
    print(f"    {'iter':>5}{'RMSD':>14}")
    for i, r in enumerate(res["history"][:8]):
        print(f"    {i+1:>5}{r:>14.6f}")
    if len(res["history"]) > 8:
        print(f"    ...  final RMSD {res['history'][-1]:.2e}")

    aligned = apply_transform(res["R"], res["t"], src)
    final = math.sqrt(_mean_sq_error(aligned, shuffled, _nearest_neighbours(aligned, shuffled)))
    print(f"\n  recovered rotation angle: {math.atan2(res['R'][1][0], res['R'][0][0]):.4f} rad "
          f"(true {theta})")
    print(f"  final alignment RMSD: {final:.2e}  ->  clouds registered")
    print(f"\n  ICP alternates: match each point to its nearest neighbour, solve the optimal rigid")
    print(f"  transform (Kabsch), repeat. Each round can only lower the error, so it converges.")

    _svg(os.path.join(outdir, "icp.svg"), src, shuffled, aligned)
    print(f"\n  wrote {os.path.join(outdir, 'icp.svg')}")


def _svg(path, src, target, aligned, width=760, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'ICP: source (gray), target (orange), ICP-aligned source (blue, snapped onto target)</text>',
    ]
    allpts = src + target + aligned
    xs = [p[0] for p in allpts]
    ys = [p[1] for p in allpts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span = max(maxx - minx, maxy - miny) * 1.15
    cx0, cy0 = (minx + maxx) / 2, (miny + maxy) / 2
    ox, oy, sz = 60, 50, min(width - 120, height - 100)

    def sx(x):
        return ox + sz / 2 + sz * (x - cx0) / span

    def sy(y):
        return oy + sz / 2 - sz * (y - cy0) / span

    for p, col, r in [(src, "#8b949e", 4), (target, "#ff922b", 6), (aligned, "#4dabf7", 3)]:
        for pt in p:
            parts.append(f'<circle cx="{sx(pt[0]):.1f}" cy="{sy(pt[1]):.1f}" r="{r}" fill="{col}" '
                         f'fill-opacity="0.8"/>')
    parts.append(f'<text x="{ox}" y="{height-16}" fill="#8b949e" font-size="10">'
                 f'blue lands on orange = ICP recovered the unknown rotation + translation from '
                 f'shuffled points</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
