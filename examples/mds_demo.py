"""Demo: classical MDS rebuilding a map from a distance table.

Given only the pairwise distances between "cities", classical multidimensional scaling reconstructs
their layout -- shown by recovering a known arrangement (distances in, coordinates out) and aligning
it back onto the truth with Procrustes. Also shows the eigenvalue scree revealing that a flat map
lives in two dimensions. Draws the true vs recovered positions.

    python examples/mds_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mds import (distance_matrix, classical_mds, stress, eigenvalue_spectrum,  # noqa: E402
                 procrustes_align, reconstructed_distances)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    # a rough "map" of cities (only the distance table will be given to MDS)
    cities = {
        "A": (0.0, 0.0), "B": (4.0, 0.5), "C": (5.0, 3.5),
        "D": (2.0, 4.0), "E": (0.5, 2.5), "F": (2.5, 2.0),
    }
    names = list(cities)
    truth = [cities[n] for n in names]
    D = distance_matrix(truth)

    print("Classical MDS: reconstructing a map from a distance table\n")
    print(f"  {len(names)} cities; input is ONLY the {len(names)}x{len(names)} distance matrix.\n")
    print("  distance table (rounded):")
    print("        " + "".join(f"{n:>6}" for n in names))
    for i, n in enumerate(names):
        print(f"    {n:>3} " + "".join(f"{D[i][j]:6.2f}" for j in range(len(names))))

    coords, vals = classical_mds(D, 2)
    print(f"\n  Recovered a 2-D embedding; stress (distance mismatch) = {stress(D, coords):.2e}")
    Drec = reconstructed_distances(coords)
    max_err = max(abs(D[i][j] - Drec[i][j]) for i in range(len(names)) for j in range(len(names)))
    print(f"  max pairwise-distance error after reconstruction = {max_err:.2e}")

    spec = eigenvalue_spectrum(D)
    print("\n  Eigenvalue scree (how much 'shape' each axis carries -- flat map => 2 positive):")
    for i, v in enumerate(spec):
        bar = "#" * int(round(max(v, 0) / spec[0] * 30))
        print(f"    axis {i + 1}: {v:8.3f} {bar}")
    n_dim = sum(1 for v in spec if v > 1e-6)
    print(f"  -> {n_dim} clearly-positive eigenvalues = intrinsic dimensionality 2\n")

    aligned = procrustes_align(truth, coords)
    print("  Procrustes-aligned recovered coordinates vs the true map:")
    print(f"    {'city':>5} {'true (x,y)':>16} {'recovered (x,y)':>18}")
    for i, n in enumerate(names):
        print(f"    {n:>5}  ({truth[i][0]:5.2f},{truth[i][1]:5.2f})    "
              f"({aligned[i][0]:5.2f},{aligned[i][1]:5.2f})")

    print("\n  Double-centering the squared-distance matrix turns distances into inner products;")
    print("  its top eigenvectors, scaled by sqrt(eigenvalue), are the coordinates. The map is")
    print("  recovered up to rotation, reflection, and translation -- distances fix shape, not")
    print("  orientation -- so Procrustes rotates the reconstruction back onto the known layout.")

    _svg(os.path.join(outdir, "mds.svg"), names, truth, aligned, spec)
    print(f"\n  wrote {os.path.join(outdir, 'mds.svg')}")


def _svg(path, names, truth, recovered, spec, width=760, height=430):
    lx0, lx1 = 45, width // 2 - 20
    y0, y1 = height - 45, 70
    allx = [p[0] for p in truth] + [p[0] for p in recovered]
    ally = [p[1] for p in truth] + [p[1] for p in recovered]
    xa, xb = min(allx) - 0.5, max(allx) + 0.5
    ya, yb = min(ally) - 0.5, max(ally) + 0.5

    def LX(x):
        return lx0 + (x - xa) / (xb - xa) * (lx1 - lx0)

    def LY(y):
        return y0 - (y - ya) / (yb - ya) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Classical MDS: map recovered from distances alone</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'left: true cities (green) vs MDS reconstruction (blue), Procrustes-aligned; right: '
        f'eigenvalue scree</text>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.1"/>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.1"/>',
    ]
    # connect true->recovered with a faint line, then draw both
    for i in range(len(names)):
        parts.append(f'<line x1="{LX(truth[i][0]):.1f}" y1="{LY(truth[i][1]):.1f}" '
                     f'x2="{LX(recovered[i][0]):.1f}" y2="{LY(recovered[i][1]):.1f}" '
                     f'stroke="#30363d" stroke-width="1"/>')
    for i, n in enumerate(names):
        parts.append(f'<circle cx="{LX(truth[i][0]):.1f}" cy="{LY(truth[i][1]):.1f}" r="4" '
                     f'fill="#06d6a0"/>')
        parts.append(f'<circle cx="{LX(recovered[i][0]):.1f}" cy="{LY(recovered[i][1]):.1f}" r="3" '
                     f'fill="#4dabf7"/>')
        parts.append(f'<text x="{LX(truth[i][0]):.1f}" y="{LY(truth[i][1])-7:.1f}" fill="#e6edf3" '
                     f'font-size="10" text-anchor="middle">{n}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{y0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">true (green) vs recovered (blue)</text>')

    # right: scree
    rx0, rx1 = width // 2 + 40, width - 30
    m = len(spec)
    bw = (rx1 - rx0) / m * 0.6
    top = max(spec[0], 1e-9)
    for i, v in enumerate(spec):
        h = max(v, 0) / top * (y0 - y1)
        cx = rx0 + i * (rx1 - rx0) / m
        col = "#ffd43b" if v > 1e-6 else "#484f58"
        parts.append(f'<rect x="{cx:.1f}" y="{y0-h:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                     f'fill="{col}"/>')
        parts.append(f'<text x="{cx+bw/2:.1f}" y="{y0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{i+1}</text>')
    parts.append(f'<line x1="{rx0}" y1="{y0}" x2="{rx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.1"/>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{y0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">eigenvalue index (2 positive = 2-D)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
