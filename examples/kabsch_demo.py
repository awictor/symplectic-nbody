"""Demo: the Kabsch algorithm -- optimally superimposing two point clouds.

Takes a point set, applies a known rotation/translation (and, for Umeyama, a scale) plus noise, then
recovers the rigid transform that best re-aligns them and reports the residual RMSD. Draws the source,
the misaligned target, and the Kabsch-aligned result.

    python examples/kabsch_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kabsch import kabsch, umeyama, apply_transform, rmsd  # noqa: E402


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)


def rot2(t):
    return [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]


def rot3(ax, ay, az):
    ca, sa = math.cos(ax), math.sin(ax)
    cb, sb = math.cos(ay), math.sin(ay)
    cc, sc = math.cos(az), math.sin(az)
    def mm(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    Rx = [[1, 0, 0], [0, ca, -sa], [0, sa, ca]]
    Ry = [[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]]
    Rz = [[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]]
    return mm(mm(Rz, Ry), Rx)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Kabsch: the optimal rotation that superimposes two point clouds\n")

    rng = LCG(2024)

    # 3D exact recovery
    pts3 = [[rng.u() * 10, rng.u() * 10, rng.u() * 10] for _ in range(12)]
    R_true = rot3(0.6, -0.9, 0.4)
    t_true = [3.0, -2.0, 5.0]
    target3 = apply_transform(R_true, t_true, pts3)
    res3 = kabsch(pts3, target3)
    print("  3D exact case (known rotation + translation applied, then recovered):")
    print(f"    RMSD before alignment: {rmsd(pts3, target3):.4f}")
    print(f"    RMSD after Kabsch:     {res3['rmsd']:.2e}  (recovered the exact transform)\n")

    # 3D with noise
    noisy = [[c[d] + 0.15 * (rng.u() - 0.5) for d in range(3)] for c in target3]
    resn = kabsch(pts3, noisy)
    print("  3D with measurement noise added to the target:")
    print(f"    RMSD after Kabsch:     {resn['rmsd']:.4f}  (best possible given the noise)\n")

    # Umeyama with scale
    s_true = 1.8
    scaled = apply_transform(R_true, t_true, pts3, scale=s_true)
    resu = umeyama(pts3, scaled, with_scale=True)
    print("  Umeyama (adds optimal scale) on a cloud scaled by 1.8x:")
    print(f"    recovered scale: {resu['scale']:.4f}  (true 1.8)")
    print(f"    RMSD after alignment: {resu['rmsd']:.2e}\n")

    print("  Center both clouds, form the cross-covariance H = P^T Q, take its SVD H = U S V^T, and the")
    print("  optimal rotation is R = V U^T -- with a determinant check so it stays a proper rotation,")
    print("  never a reflection. Closed form, no iteration; the backbone of structural alignment and")
    print("  point-cloud registration.")

    # 2D for the picture
    pts2 = [[rng.u() * 8, rng.u() * 8] for _ in range(10)]
    R2 = rot2(0.9)
    t2 = [6.0, 3.0]
    target2 = apply_transform(R2, t2, pts2)
    target2n = [[c[d] + 0.2 * (rng.u() - 0.5) for d in range(2)] for c in target2]
    res2 = kabsch(pts2, target2n)

    _svg(os.path.join(outdir, "kabsch.svg"), pts2, target2n, res2["transformed"])
    print(f"\n  wrote {os.path.join(outdir, 'kabsch.svg')}")


def _svg(path, source, target, aligned, width=760, height=420, pad=40):
    allpts = source + target + aligned
    xs = [p[0] for p in allpts]
    ys = [p[1] for p in allpts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    sx = (width - 2 * pad) / (xmax - xmin or 1)
    sy = (height - 2 * pad) / (ymax - ymin or 1)
    s = min(sx, sy)

    def px(x):
        return pad + (x - xmin) * s

    def py(y):
        return height - pad - (y - ymin) * s

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'Kabsch alignment: source rotated onto the noisy target</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'blue: source     orange: target (rotated + noisy)     green: source after Kabsch</text>',
    ]

    def draw(points, colour, r):
        out = []
        for p in points:
            out.append(f'<circle cx="{px(p[0]):.1f}" cy="{py(p[1]):.1f}" r="{r}" fill="{colour}"/>')
        return out

    parts += draw(source, "#4dabf7", 4)
    parts += draw(target, "#ff922b", 4)
    parts += draw(aligned, "#06d6a0", 3)
    # connect aligned -> target to show residuals
    for a, t in zip(aligned, target):
        parts.append(f'<line x1="{px(a[0]):.1f}" y1="{py(a[1]):.1f}" x2="{px(t[0]):.1f}" '
                     f'y2="{py(t[1]):.1f}" stroke="#8b949e" stroke-width="0.6" opacity="0.6"/>')
    parts.append(f'<text x="{pad}" y="{height-14}" fill="#8b949e" font-size="11">'
                 f'green lands on orange -- the residual gaps are just the added noise</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
