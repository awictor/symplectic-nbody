"""Demo: L-BFGS quasi-Newton optimization, and why it crushes gradient descent.

Minimizes the Rosenbrock function and an ill-conditioned quadratic, fits a logistic regression, and
compares L-BFGS to plain gradient descent -- drawing both convergence curves.

    python examples/lbfgs_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lbfgs import minimize  # noqa: E402


def rosen(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


def rosen_g(x):
    n = len(x)
    g = [0.0] * n
    for i in range(n - 1):
        g[i] += -400 * x[i] * (x[i + 1] - x[i] ** 2) - 2 * (1 - x[i])
        g[i + 1] += 200 * (x[i + 1] - x[i] ** 2)
    return g


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("L-BFGS: limited-memory quasi-Newton optimization\n")

    # --- Rosenbrock ---
    hist_lbfgs = []
    def rb_track(x):
        v = rosen(x)
        hist_lbfgs.append(v)
        return v
    r = minimize(rb_track, [-1.2, 1.0], grad=rosen_g, max_iter=2000)
    print(f"  Rosenbrock (2D banana):")
    print(f"    L-BFGS: fx = {r['fx']:.3e} at ({r['x'][0]:.5f}, {r['x'][1]:.5f}) "
          f"in {r['iterations']} iterations")

    # --- ill-conditioned quadratic: L-BFGS vs gradient descent ---
    scales = [1.0, 10.0, 100.0, 1000.0]
    def ill(x):
        return sum(scales[i] * x[i] ** 2 for i in range(len(x)))
    def ill_g(x):
        return [2 * scales[i] * x[i] for i in range(len(x))]

    x0 = [1.0, 1.0, 1.0, 1.0]
    lb_hist = []
    def ill_track(x):
        lb_hist.append(ill(x))
        return ill(x)
    rl = minimize(ill_track, x0, grad=ill_g, max_iter=100)

    # gradient descent, must use a small step for stability on the stiff direction
    gd_hist = []
    x = list(x0)
    lr = 1.0 / (2 * max(scales))
    for _ in range(100):
        gd_hist.append(ill(x))
        g = ill_g(x)
        x = [x[i] - lr * g[i] for i in range(len(x))]

    print(f"\n  Ill-conditioned quadratic (condition number 1000):")
    print(f"    L-BFGS after {len(lb_hist)} evals:          {rl['fx']:.3e}")
    print(f"    gradient descent after 100 steps:  {gd_hist[-1]:.3e}")
    print(f"    -> L-BFGS uses curvature to rescale each direction; GD crawls along the stiff axis.")

    # --- logistic regression ---
    state = 5
    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0
    true_w = [1.5, -2.0, 0.5]
    X, Y = [], []
    for _ in range(300):
        xi = [rng() * 4 - 2, rng() * 4 - 2]
        z = true_w[0] * xi[0] + true_w[1] * xi[1] + true_w[2]
        p = 1 / (1 + math.exp(-z))
        X.append(xi)
        Y.append(1 if rng() < p else 0)

    def sig(z):
        return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))

    def nll(w):
        t = 0.0
        for xi, y in zip(X, Y):
            p = min(max(sig(w[0] * xi[0] + w[1] * xi[1] + w[2]), 1e-12), 1 - 1e-12)
            t += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        return t + 1e-3 * sum(v * v for v in w)

    def nll_g(w):
        g = [0.0, 0.0, 0.0]
        for xi, y in zip(X, Y):
            e = sig(w[0] * xi[0] + w[1] * xi[1] + w[2]) - y
            g[0] += e * xi[0]; g[1] += e * xi[1]; g[2] += e
        return [g[0] + 2e-3 * w[0], g[1] + 2e-3 * w[1], g[2] + 2e-3 * w[2]]

    rw = minimize(nll, [0.0, 0.0, 0.0], grad=nll_g, max_iter=500)
    acc = sum(1 for xi, y in zip(X, Y)
              if (1 if rw["x"][0] * xi[0] + rw["x"][1] * xi[1] + rw["x"][2] > 0 else 0) == y) / len(Y)
    print(f"\n  Logistic regression (300 points):")
    print(f"    fitted weights [{rw['x'][0]:.2f}, {rw['x'][1]:.2f}, {rw['x'][2]:.2f}] "
          f"vs true [{true_w[0]}, {true_w[1]}, {true_w[2]}]")
    print(f"    training accuracy {acc:.2f} in {rw['iterations']} iterations")

    print("\n  L-BFGS reconstructs the action of the inverse Hessian from the last few (step,")
    print("  gradient-change) pairs via the two-loop recursion -- O(m n) memory, no matrix stored --")
    print("  giving Newton-like convergence on smooth problems that cripple first-order methods.")

    _svg(os.path.join(outdir, "lbfgs.svg"), lb_hist, gd_hist)
    print(f"\n  wrote {os.path.join(outdir, 'lbfgs.svg')}")


def _svg(path, lb_hist, gd_hist, width=760, height=430):
    m_left, m_bot, m_top, m_right = 70, 55, 80, 150
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    def logf(v):
        return math.log10(max(v, 1e-16))

    max_len = max(len(lb_hist), len(gd_hist))
    ymax = max(logf(lb_hist[0]), logf(gd_hist[0]))
    ymin = -16

    def px(i):
        return m_left + i / max_len * pw

    def py(v):
        return m_top + ph - (logf(v) - ymin) / (ymax - ymin) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'L-BFGS vs gradient descent on an ill-conditioned quadratic</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'objective (log scale) vs iteration; L-BFGS uses curvature, GD crawls the stiff axis</text>',
    ]

    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    for e in range(-16, int(ymax) + 1, 4):
        y = m_top + ph - (e - ymin) / (ymax - ymin) * ph
        parts.append(f'<text x="{m_left-8}" y="{y+4:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="end">1e{e}</text>')
    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height-12}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">iteration</text>')

    for hist, col, name in ((lb_hist, "#4dabf7", "L-BFGS"), (gd_hist, "#ff6b6b", "gradient descent")):
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(hist))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')

    ly = m_top + 10
    for col, name in (("#4dabf7", "L-BFGS"), ("#ff6b6b", "gradient descent")):
        parts.append(f'<line x1="{m_left+pw+12}" y1="{ly}" x2="{m_left+pw+30}" y2="{ly}" '
                     f'stroke="{col}" stroke-width="2.5"/>')
        parts.append(f'<text x="{m_left+pw+34}" y="{ly+4}" fill="#e6edf3" font-size="11">{name}</text>')
        ly += 20

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
