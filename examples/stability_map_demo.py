"""Demo: render a three-body stability map -- escape time over a grid of ICs.

Each pixel is a full three-body integration; its colour is how long the system
stayed bound before a body escaped. Bright = long-lived (near-periodic or deeply
bound), dark = quickly ionized. The boundary between them is fractal: the
signature of chaos in initial-condition space.

Rows are computed in parallel across CPU cores.

    python examples/stability_map_demo.py [output_dir] [grid_n]
"""

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stability_map import scan_row  # noqa: E402

_BARS = " .:-=+*#@"


def _viridis_like(t):
    """Cheap perceptual-ish colormap: dark blue -> teal -> green -> yellow."""
    t = max(0.0, min(1.0, t))
    stops = [(0.02, 0.02, 0.15), (0.13, 0.30, 0.55), (0.12, 0.57, 0.55),
             (0.42, 0.75, 0.30), (0.99, 0.91, 0.15)]
    seg = t * (len(stops) - 1)
    i = min(int(seg), len(stops) - 2)
    f = seg - i
    a, b = stops[i], stops[i + 1]
    r = a[0] + (b[0] - a[0]) * f
    g = a[1] + (b[1] - a[1]) * f
    bl = a[2] + (b[2] - a[2]) * f
    return f"#{int(r*255):02x}{int(g*255):02x}{int(bl*255):02x}"


def _worker(args):
    j, xs, y, dt, t_max = args
    return j, scan_row(j, xs, y, dt, t_max)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    os.makedirs(outdir, exist_ok=True)

    extent, dt, t_max = 1.5, 0.01, 30.0
    xs = [-extent + 2 * extent * i / (n - 1) for i in range(n)]
    ys = [-extent + 2 * extent * j / (n - 1) for j in range(n)]

    print(f"Three-body stability map: {n}x{n} = {n*n} full integrations")
    print("computing rows in parallel...")
    jobs = [(j, xs, ys[j], dt, t_max) for j in range(n)]
    grid = [None] * n
    with ProcessPoolExecutor() as ex:
        for j, row in ex.map(_worker, jobs):
            grid[j] = row

    # ASCII preview
    print("\nescape-time map (bright/@=long-lived, dark/space=quick escape):")
    for j in range(n - 1, -1, -max(1, n // 30)):  # top-down, downsampled
        line = "".join(
            _BARS[min(8, int(grid[j][i] / t_max * 8))]
            for i in range(0, n, max(1, n // 60)))
        print("  " + line)

    # SVG heatmap
    size, pad = 720, 40
    cell = (size - 2 * pad) / n
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    for j in range(n):
        py = size - pad - (j + 1) * cell  # flip y up
        for i in range(n):
            col = _viridis_like(grid[j][i] / t_max)
            px = pad + i * cell
            parts.append(f'<rect x="{px:.2f}" y="{py:.2f}" width="{cell:.2f}" '
                         f'height="{cell:.2f}" fill="{col}"/>')
    # mark the two primaries
    def sx(x): return pad + (x + extent) / (2 * extent) * (size - 2 * pad)
    def sy(y): return size - (pad + (y + extent) / (2 * extent) * (size - 2 * pad))
    for px in (-0.5, 0.5):
        parts.append(f'<circle cx="{sx(px):.1f}" cy="{sy(0):.1f}" r="4" '
                     f'fill="none" stroke="#ffffff" stroke-width="1.5"/>')
    parts.append(f'<text x="{pad}" y="28" fill="#e6edf3" font-size="18">'
                 f'Three-body stability map</text>')
    parts.append(f'<text x="{pad}" y="{size-16}" fill="#8b949e" font-size="11">'
                 f'colour = time until a body escapes; circles = two primaries; '
                 f'fractal edge = chaos</text>')
    parts.append("</svg>")
    path = os.path.join(outdir, "stability_map.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"\nwrote {path}")

    bound = sum(1 for j in range(n) for i in range(n) if grid[j][i] >= t_max)
    print(f"{bound}/{n*n} initial conditions stayed bound for the full {t_max} time units.")


if __name__ == "__main__":
    main()
