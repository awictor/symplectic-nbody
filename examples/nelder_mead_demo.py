"""Demo: Nelder-Mead simplex optimization, derivative-free.

Minimizes the Rosenbrock banana and other benchmarks using function values only, tracing the
best-vertex path as the simplex crawls down the curved valley and showing the value plunging with
no gradient. Contrasts with the derivative-based Newton method (fewer iterations, but needs a
Jacobian).

    python examples/nelder_mead_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nelder_mead import minimize, minimize_restart, sphere, rosenbrock, beale  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Nelder-Mead: derivative-free minimization by a crawling simplex\n")
    print("  Uses ONLY function values -- no gradient, no Jacobian. A simplex of n+1 points")
    print("  reflects, expands, contracts, and shrinks its way downhill.\n")

    for name, fn, start, opt in [("Sphere", sphere, [3.0, -2.0], "(0,0)"),
                                 ("Rosenbrock", rosenbrock, [-1.2, 1.0], "(1,1)"),
                                 ("Beale", beale, [1.0, 1.0], "(3,0.5)")]:
        p, v, h = minimize(fn, start, track=True)
        print(f"  {name:>10} from {start}: min {v:.2e} at "
              f"({', '.join(f'{x:.4f}' for x in p)}) [true min at {opt}]  "
              f"{h['n_iter']} iters, {h['n_eval']} evals")

    # convergence trace on Rosenbrock
    _, _, hist = minimize(rosenbrock, [-1.2, 1.0], track=True)
    print("\n  Rosenbrock best-vertex value as the simplex crawls the banana valley:")
    hs = hist["history"]
    for it in (0, 5, 20, 50, 100, len(hs) - 1):
        if it < len(hs):
            print(f"    iter {it:>3}: {hs[it]:.3e}")

    # restart refinement
    p_one, v_one = minimize(rosenbrock, [2.0, 2.0])
    p_re, v_re = minimize_restart(rosenbrock, [2.0, 2.0], restarts=3)
    print(f"\n  Restarting the simplex refines the result: single run {v_one:.2e} -> "
          f"3 restarts {v_re:.2e}")

    # dimensional note
    print("\n  Cost of no derivatives -- function evaluations grow with dimension (Sphere):")
    for dim in (2, 4, 8, 16):
        _, _, h = minimize(sphere, [1.0] * dim, track=True)
        print(f"    {dim:>2}-D: {h['n_eval']:>5} evaluations to converge")

    print("\n  The simplex tumbles and stretches like an amoeba: reflecting the worst vertex")
    print("  through the others, expanding into promising directions, contracting when it")
    print("  overshoots, shrinking when stuck. Robust and gradient-free -- the default when all")
    print("  you can do is evaluate the function.")

    _svg(os.path.join(outdir, "nelder_mead.svg"), hist["history"])
    print(f"\n  wrote {os.path.join(outdir, 'nelder_mead.svg')}")


def _svg(path, history, width=760, height=400):
    lx0, lx1 = 60, width - 40
    y0, y1 = height - 55, 70
    n = len(history)
    hs = [max(h, 1e-14) for h in history]
    lo = math.log10(min(hs))
    hi = math.log10(max(hs))

    def X(k):
        return lx0 + k / max(1, n - 1) * (lx1 - lx0)

    def Y(v):
        return y0 - (math.log10(max(v, 1e-14)) - lo) / (hi - lo) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Nelder-Mead on Rosenbrock: best-vertex value (log scale)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'the simplex crawls the curved banana valley using only function values -- no '
        f'gradient</text>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>',
    ]
    p = int(math.floor(lo))
    while p <= hi:
        yy = Y(10 ** p)
        parts.append(f'<line x1="{lx0}" y1="{yy:.1f}" x2="{lx1}" y2="{yy:.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{lx0-6:.1f}" y="{yy+4:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">1e{p}</text>')
        p += 3
    pts = " ".join(f"{X(k):.1f},{Y(hs[k]):.1f}" for k in range(n))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="2.2"/>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{y0+34:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">iteration</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
