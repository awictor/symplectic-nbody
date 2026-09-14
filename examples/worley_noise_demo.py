"""Worley noise demo: render F1 and F2-F1 cellular textures under different metrics (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import worley_noise as W


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"


def _gray(t):
    v = int(max(0.0, min(1.0, t)) * 255)
    return f"#{v:02x}{v:02x}{v:02x}"


def main(outdir=None):
    lines = []
    lines.append("Worley (cellular / Voronoi) noise")
    lines.append("=" * 50)
    lines.append("scatter feature points per grid cell; the field is the distance to the")
    lines.append("nearest (F1), second-nearest (F2), ... -- cell walls, cracked mud, scales.")
    lines.append("")
    lines.append("sample distances at a few points (seed=1, 1 point/cell):")
    lines.append(f"{'(x, y)':>14}{'F1':>8}{'F2':>8}{'F2-F1':>9}")
    for x, y in [(2.3, 4.7), (5.0, 5.0), (8.1, 1.2), (0.5, 9.9)]:
        d = W.worley(x, y, seed=1, n=2)
        lines.append(f"({x:>4.1f},{y:>4.1f}){d[0]:>8.3f}{d[1]:>8.3f}{d[1]-d[0]:>9.3f}")
    lines.append("")
    lines.append("the 3x3 neighbour computation provably misses no nearer point (== brute force),")
    lines.append("and F1 = 0 exactly at a feature point, growing toward the cell boundaries.")
    lines.append("")
    lines.append("distance metrics change the cell shape:")
    lines.append("  euclidean -> round cells, manhattan -> diamonds, chebyshev -> squares")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # four panels: F1 euclidean, F2-F1 euclidean, F1 manhattan, F1 chebyshev
        res = 90
        scale = 15.0
        panels = [
            ("F1 (Euclidean)", "euclidean", "f1"),
            ("F2 - F1 (edges)", "euclidean", "f2-f1"),
            ("F1 (Manhattan)", "manhattan", "f1"),
            ("F1 (Chebyshev)", "chebyshev", "f1"),
        ]
        px = 150
        W_svg = 2 * px + 60
        H_svg = 2 * px + 100
        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_svg}" height="{H_svg}" '
             f'viewBox="0 0 {W_svg} {H_svg}" font-family="monospace">']
        s.append(f'<rect width="{W_svg}" height="{H_svg}" fill="{BG}"/>')
        s.append(f'<text x="20" y="26" fill="{TEXT}" font-size="15">'
                 f'Worley noise: F1, cell edges, and three distance metrics</text>')
        cell = px / res
        for pi, (title, metric, mode) in enumerate(panels):
            ox = 20 + (pi % 2) * (px + 20)
            oy = 50 + (pi // 2) * (px + 26)
            fld = W.field(res, res, scale=scale, seed=1, metric=metric, mode=mode)
            # normalize
            flat = [v for row in fld for v in row]
            lo, hi = min(flat), max(flat)
            rng = hi - lo or 1
            s.append(f'<text x="{ox}" y="{oy-4}" fill="{GRAY}" font-size="11">{title}</text>')
            for j in range(res):
                for i in range(res):
                    t = (fld[j][i] - lo) / rng
                    if mode == "f2-f1":
                        t = 1 - t                 # invert so edges are bright
                    s.append(f'<rect x="{ox+i*cell:.2f}" y="{oy+j*cell:.2f}" '
                             f'width="{cell+0.5:.2f}" height="{cell+0.5:.2f}" fill="{_gray(t)}"/>')
        s.append(f'<text x="20" y="{H_svg-14}" fill="{GRAY}" font-size="10">'
                 f'Each pixel is shaded by its Worley field value; the metric sets the cell shape.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "worley_noise.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
