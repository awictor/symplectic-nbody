"""Demo: Poisson-disk sampling -- blue noise vs clumpy uniform random points.

Generates a Poisson-disk point set with Bridson's algorithm, compares its minimum spacing and gap
coverage to a uniform-random set of the same size, and draws both side by side so the blue-noise
regularity is obvious to the eye.

    python examples/poisson_disk_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from poisson_disk import (poisson_disk_2d, min_pairwise_distance,  # noqa: E402
                          gap_fraction_2d, _LCG)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    W = H = 300.0
    R = 14.0

    print("Poisson-disk sampling: random-looking points that never crowd (blue noise)\n")

    pd = poisson_disk_2d(W, H, R, k=30, seed=2024)
    print(f"  Bridson Poisson-disk: {len(pd)} points, min distance {min_pairwise_distance(pd):.2f} "
          f"(radius {R})")
    print(f"    insertable gap fraction {gap_fraction_2d(pd, W, H, R):.4f} (near-maximal packing)\n")

    # matched-count uniform random for comparison
    rng = _LCG(2024)
    uni = [(rng.uniform(0, W), rng.uniform(0, H)) for _ in range(len(pd))]
    print(f"  Uniform random ({len(uni)} points): min distance {min_pairwise_distance(uni):.2f}")
    print(f"    -> pairs land far closer than {R}; the pattern clumps and leaves holes\n")

    print("  Bridson keeps a background grid of cells r/sqrt(2) across, so each holds at most one")
    print("  sample and a candidate is tested against only its constant-size neighbourhood -- O(n)")
    print("  overall. Every accepted point sits in the annulus [r, 2r) of an active sample, packing")
    print("  tightly while never violating the minimum distance.")

    _svg(os.path.join(outdir, "poisson_disk.svg"), pd, uni, W, H, R)
    print(f"\n  wrote {os.path.join(outdir, 'poisson_disk.svg')}")


def _svg(path, pd, uni, W, H, R, pad=30, gap=40):
    scale = 0.9
    panel = W * scale
    width = int(pad * 2 + panel * 2 + gap)
    height = int(pad + 40 + panel + 30)

    def panel_svg(points, ox, oy, title, colour):
        out = [f'<text x="{ox:.0f}" y="{oy-10:.0f}" fill="#e6edf3" font-size="14">{title}</text>',
               f'<rect x="{ox:.0f}" y="{oy:.0f}" width="{panel:.0f}" height="{panel:.0f}" '
               f'fill="#010409" stroke="#30363d" stroke-width="1"/>']
        for x, y in points:
            cx = ox + x * scale
            cy = oy + y * scale
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.4" fill="{colour}"/>')
        return out

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="{pad}" y="24" fill="#e6edf3" font-size="17">'
        f'Poisson-disk (blue noise) vs uniform random -- same point count</text>',
    ]
    oy = pad + 40
    parts += panel_svg(pd, pad, oy, "Poisson-disk: even spacing, no clumps", "#06d6a0")
    parts += panel_svg(uni, pad + panel + gap, oy, "Uniform random: clumps and holes", "#ff6b6b")
    parts.append(f'<text x="{pad}" y="{height-10}" fill="#8b949e" font-size="11">'
                 f'both sets have {len(pd)} points; only the left maintains a guaranteed minimum '
                 f'distance of {R:.0f}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
