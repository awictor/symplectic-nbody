"""Demo: Johnson-Lindenstrauss random projection -- shrinking dimension, keeping distances.

Projects high-dimensional random points down to a handful of dimensions and shows that pairwise
distances survive, that distortion shrinks as the target dimension grows, and that the sparse
Achlioptas variant matches the Gaussian one. Draws the distortion-vs-dimension curve and a
projected-vs-original distance scatter.

    python examples/johnson_lindenstrauss_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from johnson_lindenstrauss import (  # noqa: E402
    min_dimension,
    gaussian_matrix,
    achlioptas_matrix,
    project,
    max_distortion,
    mean_sq_ratio,
)


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

    print("Johnson-Lindenstrauss: project to O(log n / eps^2) dims, distances survive\n")

    rng = _lcg(2024)
    n, orig_dim = 60, 500
    data = [[rng() * 2 - 1 for _ in range(orig_dim)] for _ in range(n)]
    print(f"  {n} points in {orig_dim} dimensions.\n")

    print(f"  {'target dim':>10}  {'gaussian distortion':>20}  {'achlioptas':>11}")
    dims = [10, 25, 50, 100, 200, 400]
    curve_g = []
    for k in dims:
        Mg = gaussian_matrix(k, orig_dim, seed=7)
        Ma = achlioptas_matrix(k, orig_dim, seed=7)
        dg = max_distortion(data, project(data, Mg))
        da = max_distortion(data, project(data, Ma))
        curve_g.append((k, dg))
        print(f"  {k:>10}  {dg:>20.3f}  {da:>11.3f}")

    print("\n  Worst-case pairwise distance distortion shrinks like 1/sqrt(target_dim).")
    print(f"  The JL bound for n={n}, eps=0.25 says target dim >= {min_dimension(n, 0.25)} suffices")
    print("  for (1 +/- eps) distortion on ALL pairs -- independent of the original 500 dimensions.")

    # verify unbiasedness at one dimension
    k = 100
    proj = project(data, gaussian_matrix(k, orig_dim, seed=3))
    print(f"\n  At target dim {k}: mean squared-distance ratio {mean_sq_ratio(data, proj):.4f} "
          f"(unbiased, ~1).")

    _svg(os.path.join(outdir, "johnson_lindenstrauss.svg"), data, curve_g, orig_dim)
    print(f"\n  wrote {os.path.join(outdir, 'johnson_lindenstrauss.svg')}")


def _svg(path, data, curve, orig_dim, width=760, height=430):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Left: distortion falls as target dim grows. Right: projected vs original distances</text>',
    ]

    # ---- left: distortion vs dim (log-x) ------------------------------------------------
    ox, oy, ow, oh = 55, 60, 300, 300
    lx = [math.log10(k) for k, _ in curve]
    ys = [d for _, d in curve]
    lxmin, lxmax = min(lx), max(lx)
    ymax = max(ys)

    def px(v):
        return ox + ow * (v - lxmin) / (lxmax - lxmin or 1)

    def py(v):
        return oy + oh * (1 - v / (ymax or 1))

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    pts = " ".join(f"{px(lx[i]):.1f},{py(ys[i]):.1f}" for i in range(len(curve)))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2"/>')
    for i, (k, d) in enumerate(curve):
        parts.append(f'<circle cx="{px(lx[i]):.1f}" cy="{py(ys[i]):.1f}" r="3" fill="#ffd43b"/>')
        parts.append(f'<text x="{px(lx[i]):.1f}" y="{py(ys[i])-8:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{k}</text>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+20}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">log10 target dim</text>')
    parts.append(f'<text x="{ox-6}" y="{oy-4}" fill="#8b949e" font-size="10" text-anchor="end">'
                 f'distortion</text>')

    # ---- right: scatter of projected vs original distance -------------------------------
    from johnson_lindenstrauss import gaussian_matrix, project
    proj = project(data, gaussian_matrix(50, orig_dim, seed=9))
    pairs = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            d0 = math.sqrt(sum((a - b) ** 2 for a, b in zip(data[i], data[j])))
            d1 = math.sqrt(sum((a - b) ** 2 for a, b in zip(proj[i], proj[j])))
            pairs.append((d0, d1))
    sx0, sy0, sw, sh = 430, 60, 300, 300
    dmax = max(max(d0, d1) for d0, d1 in pairs)
    parts.append(f'<rect x="{sx0}" y="{sy0}" width="{sw}" height="{sh}" fill="none" stroke="#30363d"/>')
    # y=x reference
    parts.append(f'<line x1="{sx0}" y1="{sy0+sh}" x2="{sx0+sw}" y2="{sy0}" stroke="#ff6b6b" '
                 f'stroke-width="1" stroke-dasharray="4 3"/>')
    for d0, d1 in pairs:
        x = sx0 + sw * d0 / dmax
        y = sy0 + sh * (1 - d1 / dmax)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.6" fill="#06d6a0" opacity="0.5"/>')
    parts.append(f'<text x="{sx0+sw/2:.0f}" y="{sy0+sh+20}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">original distance</text>')
    parts.append(f'<text x="{sx0-6}" y="{sy0-4}" fill="#8b949e" font-size="10" text-anchor="end">'
                 f'projected (dim 50)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
