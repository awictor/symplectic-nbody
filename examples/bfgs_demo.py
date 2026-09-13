"""Demo: BFGS quasi-Newton optimization on the Rosenbrock banana valley.

Minimizes the notorious Rosenbrock function with BFGS and, for contrast, plain gradient descent,
showing BFGS's superlinear convergence crush the crawling gradient method. Draws the optimization path
across the banana valley's contours.

    python examples/bfgs_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bfgs import minimize  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("BFGS: quasi-Newton optimization (learns curvature from gradients)\n")

    def rosen(v):
        return (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2

    def rosen_grad(v):
        x, y = v
        return [-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)]

    start = [-1.2, 1.0]
    print(f"  Minimizing the Rosenbrock function from {start} (true min (1, 1), value 0).\n")

    # BFGS with path
    path = [list(start)]
    orig_grad = rosen_grad

    def grad_track(v):
        path.append(list(v))
        return orig_grad(v)
    res = minimize(rosen, start, grad=grad_track, tol=1e-9, max_iter=2000, track=True)

    # gradient descent for contrast
    x = list(start)
    gd_iters = 0
    for _ in range(20000):
        g = rosen_grad(x)
        if (g[0] ** 2 + g[1] ** 2) ** 0.5 < 1e-6:
            break
        x = [x[0] - 0.0015 * g[0], x[1] - 0.0015 * g[1]]
        gd_iters += 1

    print(f"  {'method':>18}  {'iterations':>11}  {'final value':>13}  {'distance to min':>16}")
    print(f"  {'BFGS':>18}  {res['iterations']:>11}  {res['fun']:>13.2e}  "
          f"{((res['x'][0]-1)**2+(res['x'][1]-1)**2)**0.5:>16.2e}")
    print(f"  {'gradient descent':>18}  {gd_iters:>11}  {rosen(x):>13.2e}  "
          f"{((x[0]-1)**2+(x[1]-1)**2)**0.5:>16.2e}")

    print(f"\n  BFGS reaches the minimum in {res['iterations']} steps by approximating the inverse")
    print(f"  Hessian; gradient descent takes ~{gd_iters} tiny steps to crawl the curved valley floor.")

    _svg(os.path.join(outdir, "bfgs.svg"), rosen, path)
    print(f"\n  wrote {os.path.join(outdir, 'bfgs.svg')}")


def _svg(path_svg, rosen, path, width=760, height=430):
    xmin, xmax = -1.5, 1.5
    ymin, ymax = -0.5, 1.5
    ox, oy, ow, oh = 40, 45, width - 80, height - 90

    def sx(x):
        return ox + ow * (x - xmin) / (xmax - xmin)

    def sy(y):
        return oy + oh * (1 - (y - ymin) / (ymax - ymin))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'BFGS path down the Rosenbrock banana valley (contours) to the minimum at (1,1)</text>',
    ]
    # contour shading: sample a grid, colour by log(f)
    gx, gy = 120, 90
    for i in range(gy):
        for j in range(gx):
            xx = xmin + (xmax - xmin) * j / (gx - 1)
            yy = ymin + (ymax - ymin) * i / (gy - 1)
            v = rosen([xx, yy])
            t = min(1.0, math.log10(v + 1) / 2.6)
            shade = int(0x0d + t * (0x40 - 0x0d))
            px = ox + ow * j / (gx - 1)
            py = oy + oh * (1 - i / (gy - 1))
            parts.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{ow/gx+1:.1f}" '
                         f'height="{oh/gy+1:.1f}" fill="rgb({shade},{shade+8},{shade})"/>')
    # the path
    pts = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in path)
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="1.8"/>')
    for i, p in enumerate(path):
        parts.append(f'<circle cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="3" fill="#ffd43b"/>')
    # minimum
    parts.append(f'<circle cx="{sx(1):.1f}" cy="{sy(1):.1f}" r="6" fill="none" stroke="#ff6b6b" '
                 f'stroke-width="2"/>')
    parts.append(f'<text x="{sx(1)+8:.0f}" y="{sy(1)-6:.0f}" fill="#ff6b6b" font-size="10">min (1,1)</text>')
    parts.append(f'<text x="{sx(path[0][0])+8:.0f}" y="{sy(path[0][1]):.0f}" fill="#ffd43b" '
                 f'font-size="10">start</text>')
    parts.append("</svg>")
    with open(path_svg, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
