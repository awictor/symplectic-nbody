"""Demo: differential evolution on standard benchmarks.

Minimizes the Sphere, Rastrigin, and Rosenbrock functions, shows the best cost falling
monotonically as the population's own spread drives the mutation scale, compares DE against random
search at equal budget, and sweeps the differential weight F. Draws the convergence curves.

    python examples/differential_evolution_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from differential_evolution import (minimize, random_search, sphere,  # noqa: E402
                                    rastrigin, rosenbrock)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Differential evolution: DE/rand/1/bin -- donor = a + F*(b - c)\n")

    benches = [("Sphere", sphere, [(-5.0, 5.0)] * 2, "origin"),
               ("Rastrigin", rastrigin, [(-5.12, 5.12)] * 2, "origin"),
               ("Rosenbrock", rosenbrock, [(-2.0, 2.0), (-1.0, 3.0)], "(1,1)")]
    curves = {}
    for name, fn, bd, opt in benches:
        best, cost, hist = minimize(fn, bd, pop_size=40, n_iter=200, F=0.7, CR=0.9,
                                    seed=1, track=True)
        curves[name] = hist["history"]
        print(f"  {name:>10} (min at {opt}): cost {cost:.6f} at "
              f"({', '.join(f'{x:.3f}' for x in best)})")

    print("\n  DE vs random search at equal budget (40x200 = 8000 evaluations):")
    for name, fn, bd, _ in benches:
        _, de = minimize(fn, bd, pop_size=40, n_iter=200, seed=5)
        _, rs = random_search(fn, bd, n_eval=40 * 200, seed=6)
        print(f"    {name:>10}: DE {de:.6f}   random {rs:.6f}")

    print("\n  Differential weight F trades exploration for refinement (Rastrigin, 60 iters):")
    for F in (0.3, 0.5, 0.7, 0.9):
        _, _, h = minimize(rastrigin, [(-5.12, 5.12)] * 2, pop_size=40, n_iter=60,
                           F=F, seed=3, track=True)
        print(f"    F = {F}: best cost after 60 iters = {h['history'][-1]:.4f}")

    print("\n  Scales to higher dimensions (Sphere):")
    for dim in (2, 5, 10, 20):
        _, c = minimize(sphere, [(-5.0, 5.0)] * dim, pop_size=max(20, 5 * dim),
                        n_iter=400, seed=7)
        print(f"    {dim:>2}-D: cost {c:.6f}")

    print("\n  DE mutates by adding a SCALED DIFFERENCE between population members, so the")
    print("  population's own spread sets the step size -- large while dispersed (exploring), small")
    print("  as it converges (refining), with no schedule to tune. Greedy selection keeps the best")
    print("  monotone. It is one of the most robust black-box optimizers for continuous problems.")

    _svg(os.path.join(outdir, "differential_evolution.svg"), curves)
    print(f"\n  wrote {os.path.join(outdir, 'differential_evolution.svg')}")


def _svg(path, curves, width=760, height=410):
    colors = {"Sphere": "#4dabf7", "Rastrigin": "#ffd43b", "Rosenbrock": "#ff6b6b"}
    lx0, lx1 = 60, width - 140
    y0, y1 = height - 55, 70
    n = max(len(c) for c in curves.values())
    all_v = [max(v, 1e-12) for c in curves.values() for v in c]
    lo = math.log10(min(all_v))
    hi = math.log10(max(all_v))

    def X(k):
        return lx0 + k / (n - 1) * (lx1 - lx0)

    def Y(v):
        return y0 - (math.log10(max(v, 1e-12)) - lo) / (hi - lo) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Differential evolution: best cost per generation (log scale)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'greedy selection makes the best-so-far monotone; all three benchmarks driven to their '
        f'global minimum</text>',
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
    for name, c in curves.items():
        pts = " ".join(f"{X(k):.1f},{Y(c[k]):.1f}" for k in range(len(c)))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[name]}" '
                     f'stroke-width="2.2"/>')
        parts.append(f'<text x="{lx1+8:.1f}" y="{Y(c[-1])+3:.1f}" fill="{colors[name]}" '
                     f'font-size="11">{name}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{y0+34:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">generation</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
