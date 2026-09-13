"""Demo: mean-shift clustering -- discovering the number of clusters by seeking density modes.

Clusters a set of blobs without being told how many there are, shows how the bandwidth controls the
cluster count, and draws the points coloured by cluster with the discovered modes marked and a couple
of mode-climb trajectories.

    python examples/mean_shift_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mean_shift import mean_shift, climb, n_clusters, _mean_shift_vector, _gaussian_kernel  # noqa: E402


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def _gauss(rng):
    u1 = max(rng(), 1e-12)
    u2 = rng()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Mean-shift clustering: no k needed -- it finds the modes of the density itself\n")

    rng = _lcg(2024)
    centers = [(0, 0), (9, 1), (4, 8), (10, 9)]
    pts = []
    for c in centers:
        for _ in range(25):
            pts.append([c[0] + 0.7 * _gauss(rng), c[1] + 0.7 * _gauss(rng)])

    print(f"  {len(pts)} points drawn from {len(centers)} true blobs (not told to the algorithm).\n")
    print(f"  {'bandwidth':>10}  {'clusters found':>15}")
    for bw in [0.6, 1.0, 2.0, 4.0, 8.0]:
        labels, modes = mean_shift(pts, bandwidth=bw)
        marker = "  <- matches truth" if n_clusters(labels) == len(centers) else ""
        print(f"  {bw:>10.1f}  {n_clusters(labels):>15}{marker}")

    bw = 2.0
    labels, modes = mean_shift(pts, bandwidth=bw)
    print(f"\n  At bandwidth {bw}: {len(modes)} clusters, modes at:")
    for m in modes:
        print(f"    ({m[0]:.2f}, {m[1]:.2f})")
    print("\n  Each point slides uphill along the density gradient to the nearest peak; points that")
    print("  reach the same peak are one cluster. The bandwidth is the only knob -- no preset k.")

    # trace a couple of climbs
    trajectories = []
    for seed_pt in [pts[0], pts[30], pts[60]]:
        traj = [list(seed_pt)]
        x = list(seed_pt)
        for _ in range(40):
            x = _mean_shift_vector(x, pts, bw, _gaussian_kernel)
            traj.append(list(x))
        trajectories.append(traj)

    _svg(os.path.join(outdir, "mean_shift.svg"), pts, labels, modes, trajectories)
    print(f"\n  wrote {os.path.join(outdir, 'mean_shift.svg')}")


def _svg(path, pts, labels, modes, trajectories, width=760, height=430):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs) - 1, max(xs) + 1
    ymin, ymax = min(ys) - 1, max(ys) + 1
    ox, oy, ow, oh = 40, 50, width - 80, height - 90

    def sx(x):
        return ox + ow * (x - xmin) / (xmax - xmin)

    def sy(y):
        return oy + oh * (1 - (y - ymin) / (ymax - ymin))

    cols = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#b197fc", "#ff6b6b"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Mean-shift: {len(modes)} clusters discovered (points coloured by mode)</text>',
    ]
    for i, p in enumerate(pts):
        col = cols[labels[i] % len(cols)]
        parts.append(f'<circle cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="3" fill="{col}" '
                     f'opacity="0.7"/>')
    # trajectories
    for traj in trajectories:
        pts_str = " ".join(f"{sx(q[0]):.1f},{sy(q[1]):.1f}" for q in traj)
        parts.append(f'<polyline points="{pts_str}" fill="none" stroke="#8b949e" stroke-width="1" '
                     f'stroke-dasharray="2 2"/>')
    # modes
    for m in modes:
        parts.append(f'<circle cx="{sx(m[0]):.1f}" cy="{sy(m[1]):.1f}" r="7" fill="none" '
                     f'stroke="#e6edf3" stroke-width="2.5"/>')
    parts.append('<text x="20" y="%d" fill="#8b949e" font-size="10">'
                 'white rings = discovered modes; dashed lines = mode-climb trajectories</text>'
                 % (height - 14))
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
