"""Demo: SVD and PCA -- the axes a matrix acts along.

Computes the SVD of a matrix (checking A = U S V^T), reads off its rank / norm / condition
number, shows the best low-rank approximation shrinking the error, and runs PCA on a tilted 2D
cloud to recover its principal axes. Draws the data cloud with its principal components and the
singular-value spectrum.

    python examples/svd_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from svd import (svd, reconstruct, rank, spectral_norm, condition_number,  # noqa: E402
                 low_rank_approx, pca, project, explained_variance_ratio)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    A = [[4, 0], [3, -5], [0, 2]]
    U, s, Vt = svd(A)
    print("SVD: A = U S V^T -- orthonormal axes and the stretches between them\n")
    print(f"  singular values: {[round(v, 4) for v in s]}")
    print(f"  reconstruction A = U S V^T holds: "
          f"{all(abs(reconstruct(U, s, Vt)[i][j] - A[i][j]) < 1e-4 for i in range(3) for j in range(2))}")
    print(f"  rank {rank(A)}, spectral norm {spectral_norm(A):.4f}, condition number {condition_number(A):.4f}\n")

    # low-rank approximation of a nearly-rank-1 matrix
    B = [[i * j + 0.01 * ((i * 7 + j) % 5) for j in range(1, 7)] for i in range(1, 7)]
    print("  Low-rank approximation (Eckart-Young) of a 6x6 nearly-rank-1 matrix:")
    print(f"  {'rank k':>8}{'approximation error':>22}")
    for k in (1, 2, 3, 6):
        ap = low_rank_approx(B, k)
        err = math.sqrt(sum((ap[i][j] - B[i][j]) ** 2 for i in range(6) for j in range(6)))
        print(f"  {k:>8}{err:>22.4f}")
    print("  -- rank 1 already captures almost everything: the basis of image/data compression.\n")

    # PCA on a tilted elongated cloud
    cloud = _tilted_cloud(120, seed=3)
    comps, var, mean = pca(cloud)
    ratio = explained_variance_ratio(var)
    print(f"  PCA of a tilted 2D cloud ({len(cloud)} points):")
    print(f"    component 1 {[round(c, 3) for c in comps[0]]}, explains {ratio[0]:.1%} of variance")
    print(f"    component 2 {[round(c, 3) for c in comps[1]]}, explains {ratio[1]:.1%}")
    print("\n  SVD gives the rank, the 2-norm, the condition number, the best low-rank fit, and")
    print("  the pseudo-inverse -- and PCA (SVD of centred data) finds the directions of greatest")
    print("  variance. It underlies image compression, recommender systems, and latent semantics.")

    _svg(os.path.join(outdir, "svd.svg"), cloud, comps, var, mean)
    print(f"\n  wrote {os.path.join(outdir, 'svd.svg')}")


def _tilted_cloud(n, seed=1):
    """Points elongated along [2,1] with a little spread across -- a clear principal axis."""
    state = seed
    def rnd():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    pts = []
    for _ in range(n):
        t = (rnd() - 0.5) * 10        # along the major axis
        u = (rnd() - 0.5) * 2         # across
        x = 2 * t + (-1) * u + 5
        y = 1 * t + 2 * u + 5
        pts.append([x, y])
    return pts


def _svg(path, cloud, comps, var, mean, w=760, h=400):
    xs = [p[0] for p in cloud]
    ys = [p[1] for p in cloud]
    lo = min(min(xs), min(ys)) - 1
    hi = max(max(xs), max(ys)) + 1

    # left panel: cloud + principal axes
    lx0, lx1 = 45, w // 2 - 10
    ly0, ly1 = h - 40, 65

    def LX(x):
        return lx0 + (x - lo) / (hi - lo) * (lx1 - lx0)

    def LY(y):
        return ly0 - (y - lo) / (hi - lo) * (ly0 - ly1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'PCA: the principal axes of a tilted data cloud</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'points (blue) with the 1st (green) and 2nd (orange) principal components; '
        f'singular values (right)</text>',
    ]
    for x, y in cloud:
        parts.append(f'<circle cx="{LX(x):.1f}" cy="{LY(y):.1f}" r="2" fill="#4dabf7" opacity="0.6"/>')
    # principal axes, length scaled by sqrt(variance)
    scales = [math.sqrt(v) * 2 for v in var]
    colors = ["#06d6a0", "#ff922b"]
    for c, comp in enumerate(comps):
        for sgn in (1, -1):
            ex = mean[0] + sgn * scales[c] * comp[0]
            ey = mean[1] + sgn * scales[c] * comp[1]
            parts.append(f'<line x1="{LX(mean[0]):.1f}" y1="{LY(mean[1]):.1f}" '
                         f'x2="{LX(ex):.1f}" y2="{LY(ey):.1f}" stroke="{colors[c]}" stroke-width="2.5"/>')
    parts.append(f'<circle cx="{LX(mean[0]):.1f}" cy="{LY(mean[1]):.1f}" r="3.5" fill="#ffd43b"/>')

    # right panel: singular-value bars
    rx0, rx1 = w // 2 + 45, w - 30
    ry0, ry1 = h - 55, 70
    _, s, _ = svd([[p[0] - mean[0], p[1] - mean[1]] for p in cloud])
    smax = max(s) * 1.1

    def RY(v):
        return ry0 - v / smax * (ry0 - ry1)

    slot = (rx1 - rx0) / len(s)
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    for i, sv in enumerate(s):
        cx = rx0 + (i + 0.5) * slot
        bw = slot * 0.5
        parts.append(f'<rect x="{cx-bw/2:.1f}" y="{RY(sv):.1f}" width="{bw:.1f}" '
                     f'height="{ry0-RY(sv):.1f}" fill="#8338ec"/>')
        parts.append(f'<text x="{cx:.1f}" y="{RY(sv)-5:.1f}" fill="#b197fc" font-size="10" '
                     f'text-anchor="middle">{sv:.1f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{ry0+15:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">s{i+1}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry1-6:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">singular values</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
