"""Demo: a genetic algorithm evolving solutions to three problems.

Solves the OneMax bit problem, maximizes a bumpy real function, and packs a 0/1 knapsack -- showing
the population's best and average fitness climbing each generation, and that elitism keeps the best
monotone. Illustrates optimization by selection, crossover, and mutation, with no gradients.

    python examples/genetic_algorithm_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from genetic_algorithm import (genetic_algorithm, binary_init, binary_mutate,  # noqa: E402
                               real_init, real_mutate, solve_knapsack)


def bumpy(x):
    return -(x[0] ** 2 * 0.05 - 3 * math.cos(x[0]) + 3 * math.sin(1.3 * x[0]))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Genetic algorithm: optimization by simulated evolution\n")

    # --- OneMax ---
    L = 60
    best, fit, onemax_hist = genetic_algorithm(
        lambda bits: sum(bits), binary_init(L), binary_mutate(1.0 / L),
        n_generations=120, pop_size=80, seed=1, track=True)
    print(f"  OneMax (maximize 1-bits in a {L}-bit genome):")
    print(f"    best fitness {fit}/{L}, reached all-ones: {fit == L}")
    print(f"    population mean fitness climbed "
          f"{onemax_hist['mean_history'][0]:.1f} -> {onemax_hist['mean_history'][-1]:.1f}\n")

    # --- real-valued multimodal ---
    bounds = [(-20.0, 20.0)]
    best_r, fit_r, real_hist = genetic_algorithm(
        bumpy, real_init(bounds), real_mutate(1.0, bounds),
        n_generations=80, pop_size=60, seed=2, track=True)
    grid = [-20 + i * 0.01 for i in range(4000)]
    gx = max(grid, key=lambda x: bumpy([x]))
    print("  Bumpy real function (many local optima):")
    print(f"    GA found x = {best_r[0]:.3f}, fitness {fit_r:.3f}")
    print(f"    true global x = {gx:.3f}, fitness {bumpy([gx]):.3f}\n")

    # --- knapsack ---
    weights = [2, 3, 4, 5, 9, 7, 1, 6]
    values = [3, 4, 5, 8, 10, 9, 2, 7]
    cap = 20
    bits, val, wt = solve_knapsack(weights, values, cap, n_generations=200, pop_size=80, seed=3)
    chosen = [i for i in range(len(weights)) if bits[i]]
    print(f"  0/1 knapsack (capacity {cap}):")
    print(f"    chose items {chosen}, total value {val}, total weight {wt}\n")

    print("  Selection biases reproduction toward fit individuals; crossover recombines their")
    print("  genes; mutation adds new variation; elitism preserves the best. The population")
    print("  explores many basins at once -- no gradients, works on discrete or black-box problems.")

    _svg(os.path.join(outdir, "genetic_algorithm.svg"), onemax_hist, L, real_hist)
    print(f"\n  wrote {os.path.join(outdir, 'genetic_algorithm.svg')}")


def _svg(path, onemax_hist, L, real_hist, width=760, height=430):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Genetic algorithm: best and mean fitness climb each generation</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'left: OneMax (fraction of 1-bits); right: bumpy real-function fitness -- both with '
        f'elitism</text>',
    ]

    def panel(x0, x1, best_h, mean_h, ymax, title, ymin=0.0):
        py0, py1 = height - 50, 72
        n = len(best_h)

        def X(i):
            return x0 + i / (n - 1) * (x1 - x0)

        def Y(v):
            return py0 - (v - ymin) / (ymax - ymin + 1e-9) * (py0 - py1)

        cells = [
            f'<line x1="{x0}" y1="{py0}" x2="{x1}" y2="{py0}" stroke="#8b949e" stroke-width="1.1"/>',
            f'<line x1="{x0}" y1="{py0}" x2="{x0}" y2="{py1}" stroke="#8b949e" stroke-width="1.1"/>',
        ]
        bp = " ".join(f"{X(i):.1f},{Y(best_h[i]):.1f}" for i in range(n))
        mp = " ".join(f"{X(i):.1f},{Y(mean_h[i]):.1f}" for i in range(n))
        cells.append(f'<polyline points="{mp}" fill="none" stroke="#ff922b" stroke-width="1.6"/>')
        cells.append(f'<polyline points="{bp}" fill="none" stroke="#4dabf7" stroke-width="2.2"/>')
        cells.append(f'<text x="{(x0+x1)/2:.1f}" y="{py1-8:.1f}" fill="#e6edf3" font-size="12" '
                     f'text-anchor="middle">{title}</text>')
        cells.append(f'<text x="{x0+6}" y="{py1+8}" fill="#4dabf7" font-size="10">best</text>')
        cells.append(f'<text x="{x0+6}" y="{py1+22}" fill="#ff922b" font-size="10">mean</text>')
        return cells

    mid = width // 2
    parts += panel(45, mid - 15, onemax_hist["best_history"], onemax_hist["mean_history"], L,
                   "OneMax fitness")
    # real function: shift so the min plotted value is the floor
    rb = real_hist["best_history"]
    rm = real_hist["mean_history"]
    ymin = min(min(rb), min(rm))
    ymax = max(max(rb), max(rb))
    parts += panel(mid + 15, width - 20, rb, rm, ymax, "bumpy-function fitness", ymin=ymin)

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
