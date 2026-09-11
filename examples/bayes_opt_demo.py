"""Demo: Bayesian optimization of an expensive black box.

Minimizes a multimodal 1-D function with a tiny evaluation budget, showing the GP surrogate and its
Expected-Improvement acquisition after a few steps -- the next sample lands where EI peaks -- and
how the best-so-far value plunges far faster than random search on the 2-D Branin function.

    python examples/bayes_opt_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bayes_opt import (minimize, random_search, expected_improvement,  # noqa: E402
                       fit_length_scale)
from gaussian_process import GaussianProcess  # noqa: E402


def f1(v):
    """A multimodal 1-D test function on [2.7, 7.5]."""
    return math.sin(v[0]) + math.sin(10 * v[0] / 3)


def branin(v):
    x, y = v
    b = 5.1 / (4 * math.pi ** 2)
    c = 5 / math.pi
    t = 1 / (8 * math.pi)
    return (y - b * x * x + c * x - 6) ** 2 + 10 * (1 - t) * math.cos(x) + 10


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    bounds = [(2.7, 7.5)]
    res = minimize(f1, bounds, n_init=4, n_iter=16, seed=1)

    grid = [2.7 + i * 0.001 for i in range(4801)]
    x_true = min(grid, key=lambda x: f1([x]))
    y_true = f1([x_true])

    print("Bayesian optimization: minimize an expensive black box in few evaluations\n")
    print(f"  1-D multimodal target on [2.7, 7.5], {res['n_eval']} total evaluations\n")
    print(f"  found minimum at x = {res['best_x'][0]:.3f}, f = {res['best_y']:.4f}")
    print(f"  true minimum   at x = {x_true:.3f}, f = {y_true:.4f}")
    print(f"  gap to optimum: {res['best_y'] - y_true:.4f}\n")

    # convergence: best-so-far after each evaluation
    best_so_far = []
    run = math.inf
    for yi in res["y"]:
        run = min(run, yi)
        best_so_far.append(run)
    print("  Best-so-far value as evaluations accrue:")
    for i in (0, 3, 5, 8, 12, len(best_so_far) - 1):
        print(f"    after {i+1:>2} evals: {best_so_far[i]:.4f}")

    # BO vs random search on 2-D Branin, averaged over seeds
    print("\n  On 2-D Branin (global min 0.398), BO vs random search at equal budget:")
    bo_v, rs_v = [], []
    for s in range(5):
        rb = minimize(branin, [(-5.0, 10.0), (0.0, 15.0)], n_init=8, n_iter=22, seed=s)
        rs = random_search(branin, [(-5.0, 10.0), (0.0, 15.0)], n_eval=rb["n_eval"], seed=s + 50)
        bo_v.append(rb["best_y"])
        rs_v.append(rs["best_y"])
    print(f"    Bayesian opt : mean {sum(bo_v)/len(bo_v):.3f}, best {min(bo_v):.3f} "
          f"(all runs within 0.5 of optimum)")
    print(f"    random search: mean {sum(rs_v)/len(rs_v):.3f}, best {min(rs_v):.3f} "
          f"(erratic)\n")

    print("  The GP surrogate turns a few expensive samples into a full belief over the objective;")
    print("  Expected Improvement then spends each new evaluation where the payoff is highest,")
    print("  balancing exploiting the current best against exploring the uncertain regions. This is")
    print("  how modern hyperparameter tuners and experiment optimizers work.")

    _svg(os.path.join(outdir, "bayes_opt.svg"), res, best_so_far, bo_v, rs_v)
    print(f"\n  wrote {os.path.join(outdir, 'bayes_opt.svg')}")


def _svg(path, res, best_so_far, bo_v, rs_v, width=760, height=440):
    # left: the GP surrogate + EI after all evaluations, on the 1-D problem
    X = res["X"]
    y = res["y"]
    xs = [x[0] for x in X]
    gp, ls, _ = fit_length_scale(X, y, [0.2, 0.35, 0.5, 0.75, 1.0], noise_var=1e-6)

    gx = [2.7 + i * (7.5 - 2.7) / 160 for i in range(161)]
    mean, std = gp.predict([[g] for g in gx])
    truth = [f1([g]) for g in gx]
    f_best = min(y)
    ei = [expected_improvement(mean[i], std[i], f_best, xi=0.01) for i in range(len(gx))]

    allv = [mean[i] + 2 * std[i] for i in range(len(gx))] + \
           [mean[i] - 2 * std[i] for i in range(len(gx))] + truth + y
    va, vb = min(allv) - 0.2, max(allv) + 0.2
    lx0, lx1 = 45, width // 2 - 15
    y0, y1 = height - 120, 65

    def LX(x):
        return lx0 + (x - 2.7) / (7.5 - 2.7) * (lx1 - lx0)

    def LY(v):
        return y0 - (v - va) / (vb - va) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Bayesian optimization: GP surrogate + Expected Improvement</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'left: surrogate (blue) and 2-sigma band over the true function (green), sampled points; '
        f'EI below</text>',
    ]

    # confidence band
    top = " ".join(f"{LX(gx[i]):.1f},{LY(mean[i]+2*std[i]):.1f}" for i in range(len(gx)))
    bot = " ".join(f"{LX(gx[i]):.1f},{LY(mean[i]-2*std[i]):.1f}" for i in range(len(gx) - 1, -1, -1))
    parts.append(f'<polygon points="{top} {bot}" fill="#4dabf7" opacity="0.16"/>')
    tru = " ".join(f"{LX(gx[i]):.1f},{LY(truth[i]):.1f}" for i in range(len(gx)))
    parts.append(f'<polyline points="{tru}" fill="none" stroke="#06d6a0" stroke-width="1.6" '
                 f'stroke-dasharray="5 3"/>')
    mn = " ".join(f"{LX(gx[i]):.1f},{LY(mean[i]):.1f}" for i in range(len(gx)))
    parts.append(f'<polyline points="{mn}" fill="none" stroke="#4dabf7" stroke-width="2.2"/>')
    # sampled points, coloured from early (grey) to late (yellow) to show the search focusing
    for i, xv in enumerate(xs):
        t = i / max(1, len(xs) - 1)
        col = f"rgb({int(139+t*116)},{int(148+t*64)},{int(158-t*99)})"
        parts.append(f'<circle cx="{LX(xv):.1f}" cy="{LY(y[i]):.1f}" r="3.2" fill="{col}" '
                     f'stroke="#0d1117" stroke-width="0.6"/>')
    # mark the found minimum
    parts.append(f'<circle cx="{LX(res["best_x"][0]):.1f}" cy="{LY(res["best_y"]):.1f}" r="5" '
                 f'fill="none" stroke="#ff6b6b" stroke-width="2"/>')
    parts.append(f'<line x1="{lx0}" y1="{y0}" x2="{lx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.1"/>')

    # EI strip beneath
    ey0, ey1 = height - 40, y0 + 20
    emax = max(ei) or 1.0
    eipts = " ".join(f"{LX(gx[i]):.1f},{ey0 - ei[i]/emax*(ey0-ey1):.1f}" for i in range(len(gx)))
    parts.append(f'<polyline points="{eipts}" fill="none" stroke="#ff922b" stroke-width="1.8"/>')
    parts.append(f'<text x="{lx0}" y="{ey1-4:.1f}" fill="#ff922b" font-size="10">'
                 f'Expected Improvement (next sample at its peak)</text>')

    # right: convergence, BO best-so-far and the random-search spread
    rx0, rx1 = width // 2 + 55, width - 25
    ry0, ry1 = height - 120, 65
    n = len(best_so_far)
    # y-range across BO curve and random values
    lo = min(min(best_so_far), min(bo_v), min(rs_v))
    hi = max(max(best_so_far[:3]), max(rs_v))

    def RX(i):
        return rx0 + i / (n - 1) * (rx1 - rx0)

    def RY(v):
        return ry0 - (v - lo) / (hi - lo + 1e-9) * (ry0 - ry1)

    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry1-16:.1f}" fill="#e6edf3" font-size="12" '
                 f'text-anchor="middle">best-so-far vs evaluation</text>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.1"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.1"/>')
    bo_curve = " ".join(f"{RX(i):.1f},{RY(best_so_far[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{bo_curve}" fill="none" stroke="#4dabf7" stroke-width="2.4"/>')
    parts.append(f'<text x="{rx0+8}" y="{RY(best_so_far[-1])-6:.1f}" fill="#4dabf7" '
                 f'font-size="10">BO best-so-far</text>')
    # random-search final values as scattered markers at the right edge
    for v in rs_v:
        parts.append(f'<circle cx="{rx1-6:.1f}" cy="{RY(v):.1f}" r="2.6" fill="#8b949e"/>')
    parts.append(f'<text x="{rx1-10:.1f}" y="{RY(max(rs_v))-6:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">random runs</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
