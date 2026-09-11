"""Demo: ant colony optimization on a travelling-salesman tour.

Solves a random TSP instance by simulated pheromone trails, showing the best tour beating the
greedy nearest-neighbour baseline and shortening over iterations, and how pheromone concentrates on
the edges of good tours. Draws the final tour with edges shaded by pheromone strength.

    python examples/ant_colony_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ant_colony import solve, tour_length, nearest_neighbour_tour, _distance_matrix  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    state = 9

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    n = 22
    cities = [(rng() * 10, rng() * 10) for _ in range(n)]
    D = _distance_matrix(cities)

    nn = nearest_neighbour_tour(D)
    nn_len = tour_length(nn, D)
    order, length, hist = solve(cities, n_iter=120, alpha=1.0, beta=3.0, rho=0.5,
                                seed=3, track=True)

    print("Ant colony optimization: pheromone trails solving the travelling salesman\n")
    print(f"  {n} cities, {len(hist['history'])} iterations, colony of {n} ants\n")
    print(f"  nearest-neighbour greedy tour: length {nn_len:.2f}")
    print(f"  ant colony optimization:       length {length:.2f}  "
          f"({100 * (1 - length / nn_len):.0f}% shorter)\n")

    print("  Best tour length as pheromone accumulates (early rounds explore, then converge):")
    h = hist["history"]
    for it in (0, 5, 15, 40, 80, len(h) - 1):
        if it < len(h):
            print(f"    iter {it:>3}: {h[it]:.2f}")

    # pheromone concentration: how much of the total pheromone sits on the best tour's edges
    tau = hist["pheromone"]
    total_tau = sum(tau[i][j] for i in range(n) for j in range(i + 1, n))
    best_edges = set()
    for k in range(n):
        a, b = order[k], order[(k + 1) % n]
        best_edges.add((min(a, b), max(a, b)))
    tour_tau = sum(tau[i][j] for i, j in best_edges)
    print(f"\n  Pheromone on the {n} best-tour edges holds {100 * tour_tau / total_tau:.0f}% "
          f"of all pheromone (of {n * (n - 1) // 2} possible edges) -- the colony has converged.")

    # alpha/beta trade
    print("\n  Pheromone (alpha) vs greedy heuristic (beta) balance -- final tour length:")
    for a, b in [(1.0, 0.0), (1.0, 2.0), (1.0, 5.0), (0.0, 3.0)]:
        _, l = solve(cities, n_iter=80, alpha=a, beta=b, seed=3)
        label = "pheromone only" if b == 0 else ("greedy only" if a == 0 else f"a={a}, b={b}")
        print(f"    {label:>16}: {l:.2f}")

    print("\n  Each ant builds a tour choosing the next city by pheromone^alpha * (1/dist)^beta;")
    print("  pheromone evaporates then is redeposited inversely to tour length, so short tours")
    print("  reinforce their edges and the colony converges -- swarm intelligence, no central plan.")

    _svg(os.path.join(outdir, "ant_colony.svg"), cities, order, nn, tau, hist["history"])
    print(f"\n  wrote {os.path.join(outdir, 'ant_colony.svg')}")


def _svg(path, cities, order, nn_order, tau, history, width=760, height=430):
    n = len(cities)
    xs = [c[0] for c in cities]
    ys = [c[1] for c in cities]
    xa, xb = min(xs) - 0.5, max(xs) + 0.5
    ya, yb = min(ys) - 0.5, max(ys) + 0.5
    lx0, lx1 = 45, width // 2 - 20
    y0, y1 = height - 50, 70

    def LX(x):
        return lx0 + (x - xa) / (xb - xa) * (lx1 - lx0)

    def LY(y):
        return y0 - (y - ya) / (yb - ya) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Ant colony: the converged tour and pheromone (left), length over time (right)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'left: strong pheromone edges glow; the best tour is drawn bold; right: best length '
        f'falling</text>',
    ]

    # pheromone edges: draw all, opacity ~ pheromone strength
    tmax = max(tau[i][j] for i in range(n) for j in range(i + 1, n)) or 1.0
    for i in range(n):
        for j in range(i + 1, n):
            op = tau[i][j] / tmax
            if op > 0.05:
                parts.append(f'<line x1="{LX(cities[i][0]):.1f}" y1="{LY(cities[i][1]):.1f}" '
                             f'x2="{LX(cities[j][0]):.1f}" y2="{LY(cities[j][1]):.1f}" '
                             f'stroke="#ffd43b" stroke-width="{0.5 + 2 * op:.1f}" '
                             f'opacity="{0.15 + 0.6 * op:.2f}"/>')
    # best tour bold
    for k in range(n):
        a, b = order[k], order[(k + 1) % n]
        parts.append(f'<line x1="{LX(cities[a][0]):.1f}" y1="{LY(cities[a][1]):.1f}" '
                     f'x2="{LX(cities[b][0]):.1f}" y2="{LY(cities[b][1]):.1f}" '
                     f'stroke="#4dabf7" stroke-width="1.8"/>')
    for c in cities:
        parts.append(f'<circle cx="{LX(c[0]):.1f}" cy="{LY(c[1]):.1f}" r="3" fill="#ff6b6b"/>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{y0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">tour (blue) over pheromone (yellow glow)</text>')

    # right: convergence
    rx0, rx1 = width // 2 + 45, width - 30
    hh = history
    lo, hi = min(hh), max(hh)

    def RX(i):
        return rx0 + i / (len(hh) - 1) * (rx1 - rx0)

    def RY(v):
        return y0 - (v - lo) / (hi - lo + 1e-9) * (y0 - y1)

    parts.append(f'<line x1="{rx0}" y1="{y0}" x2="{rx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.1"/>')
    parts.append(f'<line x1="{rx0}" y1="{y0}" x2="{rx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.1"/>')
    pts = " ".join(f"{RX(i):.1f},{RY(hh[i]):.1f}" for i in range(len(hh)))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="2.2"/>')
    parts.append(f'<text x="{rx1-2:.1f}" y="{RY(hh[-1])-4:.1f}" fill="#06d6a0" font-size="10" '
                 f'text-anchor="end">{hh[-1]:.1f}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{y0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">iterations -> best tour length</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
