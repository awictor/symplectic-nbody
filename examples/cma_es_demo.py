"""Demo: CMA-ES minimizing hard black-box functions, with the search ellipse adapting.

Runs CMA-ES on the sphere, Rosenbrock, Rastrigin, and an ill-conditioned ellipsoid, compares to
random search, and draws the convergence curves (log fitness vs evaluations).

    python examples/cma_es_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cma_es import minimize, sphere, rosenbrock, rastrigin, ellipsoid  # noqa: E402


def _tracked(f):
    """Wrap f to record the running best fitness after each evaluation."""
    history = []
    best = [float("inf")]

    def g(x):
        v = f(x)
        if v < best[0]:
            best[0] = v
        history.append(best[0])
        return v

    return g, history


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("CMA-ES: covariance matrix adaptation for black-box minimization\n")

    problems = [
        ("sphere (4D)", sphere, [3.0, -2.0, 1.5, 0.8], 0.5, 800),
        ("Rosenbrock (2D banana)", rosenbrock, [-1.2, 1.0], 0.3, 3000),
        ("Rastrigin (2D, multimodal)", rastrigin, [0.3, -0.2], 0.3, 2000),
        ("ellipsoid (3D, cond 1e6)", ellipsoid, [1.0, 1.0, 1.0], 0.3, 3000),
    ]

    curves = {}
    for name, f, x0, s0, mi in problems:
        g, hist = _tracked(f)
        r = minimize(g, x0, sigma0=s0, max_iter=mi, seed=1)
        curves[name] = hist
        print(f"  {name:30s}: fx = {r['fx']:.3e} in {r['evaluations']} evals, x ~ "
              f"[{', '.join(f'{xi:.3f}' for xi in r['x'][:3])}{'...' if len(r['x'])>3 else ''}]")

    # random search comparison on the sphere
    def rand_search(f, x0, span, evals, seed):
        state = seed
        def u():
            nonlocal state
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            return (state >> 8) / (1 << 24)
        best = float("inf")
        n = len(x0)
        for _ in range(evals):
            best = min(best, f([x0[i] + (u() * 2 - 1) * span for i in range(n)]))
        return best

    budget = len(curves["sphere (4D)"])
    rs = rand_search(sphere, [3.0, -2.0, 1.5, 0.8], 6.0, budget, seed=1)
    print(f"\n  On the sphere, {budget} evaluations of random search reach only {rs:.3e},")
    print(f"  while CMA-ES reaches {curves['sphere (4D)'][-1]:.3e} -- a difference of many orders.")

    print("\n  CMA-ES samples from a Gaussian and, each generation, moves its mean to the best")
    print("  samples, adapts the step size from the length of its cumulative path, and bends the")
    print("  covariance toward recent progress -- learning the landscape's curvature without gradients.")

    _svg(os.path.join(outdir, "cma_es.svg"), curves)
    print(f"\n  wrote {os.path.join(outdir, 'cma_es.svg')}")


def _svg(path, curves, width=760, height=440):
    m_left, m_bot, m_top, m_right = 70, 55, 80, 180
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    # x: evaluation index (linear), y: log10(fitness) clamped
    def logf(v):
        return math.log10(max(v, 1e-16))

    max_evals = max(len(h) for h in curves.values())
    ymax = max(logf(h[0]) for h in curves.values())
    ymin = -16

    def px(i):
        return m_left + i / max_evals * pw

    def py(v):
        return m_top + ph - (logf(v) - ymin) / (ymax - ymin) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'CMA-ES convergence: best fitness (log scale) vs evaluations</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'each curve is a benchmark function; steep drops show the covariance learning the landscape</text>',
    ]

    # axes
    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    for e in range(-16, int(ymax) + 1, 4):
        y = m_top + ph - (e - ymin) / (ymax - ymin) * ph
        parts.append(f'<line x1="{m_left-4}" y1="{y:.1f}" x2="{m_left}" y2="{y:.1f}" stroke="#484f58"/>')
        parts.append(f'<text x="{m_left-8}" y="{y+4:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="end">1e{e}</text>')
    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height-12}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">function evaluations</text>')

    colors = ["#4dabf7", "#ffd43b", "#ff6b6b", "#06d6a0", "#b197fc"]
    for idx, (name, hist) in enumerate(curves.items()):
        col = colors[idx % len(colors)]
        # subsample for a smooth polyline
        step = max(1, len(hist) // 400)
        pts = " ".join(f"{px(i):.1f},{py(hist[i]):.1f}" for i in range(0, len(hist), step))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.8"/>')
        ly = m_top + 6 + idx * 18
        parts.append(f'<line x1="{m_left+pw+12}" y1="{ly}" x2="{m_left+pw+32}" y2="{ly}" '
                     f'stroke="{col}" stroke-width="2.5"/>')
        parts.append(f'<text x="{m_left+pw+36}" y="{ly+4}" fill="#e6edf3" font-size="10">{name}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
