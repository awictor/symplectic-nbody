"""Demo: percolation -- the sudden onset of connectivity.

Prints the spanning probability and largest-cluster fraction as occupation p sweeps through
the threshold, then draws the spanning-probability curve sharpening at p_c and three lattice
snapshots (below, at, above threshold) with the largest cluster highlighted.

    python examples/percolation_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from percolation import (occupy_lattice, spans, largest_cluster_fraction,  # noqa: E402
                         percolation_probability, label_clusters, _find,
                         SQUARE_SITE_THRESHOLD)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Percolation: occupy sites with probability p, look for a spanning cluster\n")
    print("  2D square-lattice site threshold p_c ~ %.4f\n" % SQUARE_SITE_THRESHOLD)
    print(f"  {'p':>6}{'spans?':>16}{'largest cluster':>18}")
    for p in (0.40, 0.50, 0.55, 0.59, 0.65, 0.75):
        pp = percolation_probability(40, p, trials=15)
        f = largest_cluster_fraction(occupy_lattice(60, p, seed=1))
        print(f"  {p:>6.2f}{pp*100:>13.0f} %{f*100:>15.0f} %")

    print("\n  Below p_c occupied sites form isolated islands; right at ~0.59 a single cluster")
    print("  abruptly spans the whole lattice -- a geometric phase transition. Same threshold")
    print("  governs forest fires, oil in rock, disease on a contact network, and random resistor grids.")

    _svg(os.path.join(outdir, "percolation.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'percolation.svg')}")


def _largest_root(grid):
    parent, n = label_clusters(grid)
    counts = {}
    best_root, best = None, 0
    for r in range(n):
        for c in range(n):
            if grid[r][c]:
                root = _find(parent, r * n + c)
                counts[root] = counts.get(root, 0) + 1
                if counts[root] > best:
                    best, best_root = counts[root], root
    return parent, best_root


def _svg(path, size=720, pad=70):
    x0, x1 = pad, size - pad
    mid = size * 0.50

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Percolation</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'spanning probability sharpens at p_c ~ 0.59 (top); lattice snapshots below/at/above (bottom)</text>',
    ]

    # --- top: spanning probability vs p ---
    ty0, ty1 = mid - 30, pad + 44
    def PX(p):
        return x0 + (p - 0.3) / (0.9 - 0.3) * (x1 - x0)
    def PY(prob):
        return ty0 - prob * (ty0 - ty1)
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x1}" y2="{ty0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x0}" y2="{ty1}" stroke="#8b949e" stroke-width="1.4"/>')
    for prob in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{x0-6:.1f}" y="{PY(prob)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{prob:.1f}</text>')
    pts = []
    p = 0.30
    while p <= 0.90:
        pts.append(f"{PX(p):.1f},{PY(percolation_probability(30, p, trials=12)):.1f}")
        p += 0.03
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')
    # p_c marker
    parts.append(f'<line x1="{PX(SQUARE_SITE_THRESHOLD):.1f}" y1="{ty0:.1f}" '
                 f'x2="{PX(SQUARE_SITE_THRESHOLD):.1f}" y2="{ty1:.1f}" '
                 f'stroke="#ff6b6b" stroke-width="1.2" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{PX(SQUARE_SITE_THRESHOLD):.1f}" y="{ty1-4:.1f}" fill="#ff6b6b" '
                 f'font-size="10" text-anchor="middle">p_c ~ 0.59</text>')
    for p in (0.3, 0.5, 0.7, 0.9):
        parts.append(f'<text x="{PX(p):.1f}" y="{ty0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{p:.1f}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{ty0+28:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">occupation probability p  (fraction of lattices that span)</text>')

    # --- bottom: three lattice snapshots ---
    labels = [(0.50, "p=0.50 (islands)"), (0.59, "p=0.59 (threshold)"), (0.72, "p=0.72 (spanning)")]
    m = 28   # lattice size to draw
    panel = (x1 - x0) / 3
    cell = (panel * 0.82) / m
    top = mid + 30
    for pi, (p, lbl) in enumerate(labels):
        grid = occupy_lattice(m, p, seed=5)
        parent, big_root = _largest_root(grid)
        gx = x0 + panel * pi + panel * 0.09
        for r in range(m):
            for c in range(m):
                if grid[r][c]:
                    root = _find(parent, r * m + c)
                    col = "#ffd43b" if root == big_root else "#4a5568"
                    parts.append(f'<rect x="{gx + c*cell:.1f}" y="{top + r*cell:.1f}" '
                                 f'width="{cell:.1f}" height="{cell:.1f}" fill="{col}"/>')
        does = pc_spans = spans(grid)
        parts.append(f'<text x="{gx + m*cell/2:.1f}" y="{top + m*cell + 16:.1f}" '
                     f'fill="#8b949e" font-size="10" text-anchor="middle">{lbl}</text>')
        parts.append(f'<text x="{gx + m*cell/2:.1f}" y="{top + m*cell + 30:.1f}" '
                     f'fill="{"#06d6a0" if does else "#ff6b6b"}" font-size="10" text-anchor="middle">'
                     f'{"spans" if does else "no spanning path"}</text>')

    parts.append(f'<text x="{x0:.1f}" y="{size-18:.1f}" fill="#ffd43b" font-size="10">'
                 f'yellow = largest cluster</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
