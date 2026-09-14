"""Cross-entropy method demo: the sampling Gaussian marching down the Rastrigin landscape (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cross_entropy_method as CEM


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
YELLOW = "#ffd43b"
RED = "#ff6b6b"


def main(outdir=None):
    lines = []
    lines.append("Cross-entropy method: derivative-free optimization")
    lines.append("=" * 52)
    lines.append("loop: sample a Gaussian, keep the best (elite) points, re-fit the Gaussian")
    lines.append("to them, repeat -- the distribution marches to the optimum and tightens.")
    lines.append("")
    lines.append("benchmark results (all minima are 0):")
    lines.append(f"{'function':>14}{'start f':>12}{'final f':>12}   minimizer")
    tests = [
        ("sphere", CEM.sphere, [5.0, 5.0, 5.0], [3] * 3, 60, 80, 0.7),
        ("Rastrigin", CEM.rastrigin, [3.0, -2.0], [4, 4], 120, 200, 0.7),
        ("Rosenbrock", CEM.rosenbrock, [-1.0, 1.0], [2, 2], 200, 400, 0.5),
    ]
    for name, f, m0, s0, pop, it, sm in tests:
        x = CEM.optimize(f, m0, std=s0, population=pop, iterations=it, smoothing=sm, seed=1)
        lines.append(f"{name:>14}{f(m0):>12.3f}{f(x):>12.5f}   {[round(v,3) for v in x]}")
    lines.append("")
    lines.append("Rastrigin is a field of local minima that traps gradient methods; CEM's")
    lines.append("population sampling walks past them to the global optimum at the origin.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # visualize CEM on 2-D Rastrigin: show the elite means over iterations on a contour-ish grid
        _, hist = CEM.optimize(CEM.rastrigin, [3.5, -3.0], std=[4, 4], population=100,
                               iterations=40, seed=3, track=True)
        # we need the mean trajectory; re-run capturing means
        means = _trace_means(CEM.rastrigin, [3.5, -3.0], [4, 4], 100, 40, 0.7, 3)

        W, H = 480, 500
        ml, mt, size = 40, 50, 400
        lo, hi = -5.0, 5.0

        def sx(x):
            return ml + (x - lo) / (hi - lo) * size

        def sy(y):
            return mt + size - (y - lo) / (hi - lo) * size

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'CEM on the Rastrigin landscape (darker = lower)</text>')
        # background: Rastrigin value as grayscale
        res = 80
        cell = size / res
        vals = []
        for j in range(res):
            row = []
            for i in range(res):
                x = lo + (i + 0.5) / res * (hi - lo)
                y = lo + (j + 0.5) / res * (hi - lo)
                row.append(CEM.rastrigin([x, y]))
            vals.append(row)
        vmax = max(max(r) for r in vals)
        for j in range(res):
            for i in range(res):
                t = vals[j][i] / vmax
                g = int(20 + t * 90)
                col = f"#{g:02x}{g:02x}{min(255, g+30):02x}"
                s.append(f'<rect x="{ml+i*cell:.2f}" y="{mt+(res-1-j)*cell:.2f}" '
                         f'width="{cell+0.6:.2f}" height="{cell+0.6:.2f}" fill="{col}"/>')
        # mean trajectory
        pts = " ".join(f"{sx(m[0]):.1f},{sy(m[1]):.1f}" for m in means)
        s.append(f'<polyline points="{pts}" fill="none" stroke="{YELLOW}" stroke-width="1.5"/>')
        for k, m in enumerate(means):
            col = RED if k == 0 else (GREEN if k == len(means) - 1 else BLUE)
            r = 5 if (k == 0 or k == len(means) - 1) else 2.5
            s.append(f'<circle cx="{sx(m[0]):.1f}" cy="{sy(m[1]):.1f}" r="{r}" fill="{col}"/>')
        # origin (true optimum)
        s.append(f'<circle cx="{sx(0):.1f}" cy="{sy(0):.1f}" r="6" fill="none" '
                 f'stroke="{GREEN}" stroke-width="2"/>')
        s.append(f'<text x="{ml}" y="{H-16}" fill="{GRAY}" font-size="10">'
                 f'The Gaussian mean (yellow path) starts at the red dot and converges to the green '
                 f'ring at the origin, stepping over the ripples of local minima.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "cross_entropy_method.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


def _trace_means(f, mean, std, population, iterations, smoothing, seed):
    """Re-run CEM capturing the mean each iteration (for visualization)."""
    rng = CEM._Rng(seed)
    dim = len(mean)
    mu = list(mean)
    sigma = list(std)
    n_elite = max(1, int(population * 0.2))
    means = [list(mu)]
    for _ in range(iterations):
        samples = []
        for _ in range(population):
            x = [mu[d] + sigma[d] * rng.normal() for d in range(dim)]
            samples.append((f(x), x))
        samples.sort(key=lambda t: t[0])
        elites = [x for _, x in samples[:n_elite]]
        new_mu = [sum(e[d] for e in elites) / n_elite for d in range(dim)]
        new_sig = [math.sqrt(max(sum((e[d] - new_mu[d]) ** 2 for e in elites) / n_elite, 1e-12))
                   for d in range(dim)]
        mu = [smoothing * new_mu[d] + (1 - smoothing) * mu[d] for d in range(dim)]
        sigma = [smoothing * new_sig[d] + (1 - smoothing) * sigma[d] for d in range(dim)]
        means.append(list(mu))
    return means


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
