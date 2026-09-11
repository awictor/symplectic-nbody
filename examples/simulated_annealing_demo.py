"""Demo: simulated annealing on a travelling-salesman tour and a multimodal function.

Solves a random travelling-salesman instance, drawing the greedy nearest-neighbour tour beside the
shorter annealed tour, and shows the cost falling as the temperature cools -- with early uphill
moves (rising cost) that let it escape local minima. Also finds the global minimum of a bumpy 1-D
function that greedy descent misses.

    python examples/simulated_annealing_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulated_annealing import (anneal, geometric_schedule, solve_tsp, tour_length,  # noqa: E402
                                 nearest_neighbour_tour, _reverse_segment, _Rng)


def f(x):
    return x[0] ** 2 * 0.05 - 3 * math.cos(x[0]) + 3 * math.sin(1.3 * x[0])


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    # --- multimodal 1-D minimization ---
    def neigh(x, rng):
        return [x[0] + rng.gauss(1.5)]

    best, cost, hist = anneal([15.0], f, neigh, geometric_schedule(10.0, 0.999), 8000,
                              seed=1, track=True)
    grid = [-20 + i * 0.01 for i in range(4000)]
    gx = min(grid, key=lambda x: f([x]))

    print("Simulated annealing: global optimization by cooling a fictitious temperature\n")
    print("  Multimodal 1-D function (many local minima):")
    print(f"    SA found x = {best[0]:.3f}, cost {cost:.3f}")
    print(f"    true global x = {gx:.3f}, cost {f([gx]):.3f}")
    print(f"    acceptance rate over the run: {hist['accept_rate']:.1%} "
          f"(high early, near zero once cold)\n")

    # --- travelling salesman ---
    rng = _Rng(9)
    n_cities = 25
    cities = [(rng.uniform() * 10, rng.uniform() * 10) for _ in range(n_cities)]
    nn = nearest_neighbour_tour(cities)
    nn_len = tour_length(nn, cities)
    sa_order, sa_len = solve_tsp(cities, steps=40000, alpha=0.9997, seed=3)

    print(f"  Travelling salesman, {n_cities} random cities:")
    print(f"    nearest-neighbour greedy tour: length {nn_len:.2f}")
    print(f"    simulated-annealing tour:      length {sa_len:.2f}  "
          f"({100*(1-sa_len/nn_len):.0f}% shorter)\n")

    # cost trajectory for the TSP, tracked
    order0 = list(range(n_cities))
    T0 = tour_length(order0, cities) / n_cities
    _, _, tsp_hist = anneal(order0, lambda o: tour_length(o, cities), _reverse_segment,
                            geometric_schedule(T0, 0.9997), 40000, seed=3, track=True)

    print("  Tour length as the system cools (early rises = escaping local minima):")
    ch = tsp_hist["cost_history"]
    for frac in (0.0, 0.05, 0.15, 0.4, 0.7, 1.0):
        i = int(frac * (len(ch) - 1))
        print(f"    {int(frac*100):>3}% through: length {ch[i]:.2f}")

    print("\n  Annealing accepts uphill moves with probability exp(-delta/T): at high T it roams")
    print("  freely and hops out of local basins; as T cools only improving moves survive and it")
    print("  settles. Cool slowly enough and it approaches the global optimum -- here a far shorter")
    print("  tour than greedy, and the true minimum of a function that traps hill-climbing.")

    _svg(os.path.join(outdir, "simulated_annealing.svg"), cities, nn, sa_order, ch,
         tsp_hist, nn_len, sa_len)
    print(f"\n  wrote {os.path.join(outdir, 'simulated_annealing.svg')}")


def _svg(path, cities, nn_order, sa_order, cost_history, hist, nn_len, sa_len,
         width=760, height=450):
    xs = [c[0] for c in cities]
    ys = [c[1] for c in cities]
    xa, xb = min(xs) - 0.5, max(xs) + 0.5
    ya, yb = min(ys) - 0.5, max(ys) + 0.5

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Simulated annealing: a shorter TSP tour (left), cost cooling (right)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'left: greedy nearest-neighbour (grey) vs annealed tour (blue); right: tour length as '
        f'temperature falls</text>',
    ]

    # left: the two tours
    lx0, lx1 = 45, width // 2 - 20
    ly0, ly1 = height - 50, 70

    def LX(x):
        return lx0 + (x - xa) / (xb - xa) * (lx1 - lx0)

    def LY(v):
        return ly0 - (v - ya) / (yb - ya) * (ly0 - ly1)

    def tour_poly(order, col, w, dash=""):
        pts = order + [order[0]]
        s = " ".join(f"{LX(cities[c][0]):.1f},{LY(cities[c][1]):.1f}" for c in pts)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{s}" fill="none" stroke="{col}" stroke-width="{w}"{d}/>'

    parts.append(tour_poly(nn_order, "#8b949e", 1.2, dash="4 3"))
    parts.append(tour_poly(sa_order, "#4dabf7", 2.0))
    for c in cities:
        parts.append(f'<circle cx="{LX(c[0]):.1f}" cy="{LY(c[1]):.1f}" r="3" fill="#ffd43b"/>')
    parts.append(f'<text x="{lx0+4}" y="{ly1-6}" fill="#8b949e" font-size="10">'
                 f'greedy {nn_len:.1f}</text>')
    parts.append(f'<text x="{lx0+4}" y="{ly1+8}" fill="#4dabf7" font-size="10">'
                 f'annealed {sa_len:.1f}</text>')

    # right: cost cooling curve (subsampled)
    rx0, rx1 = width // 2 + 40, width - 30
    ry0, ry1 = ly0, ly1 + 10
    step = max(1, len(cost_history) // 400)
    cc = cost_history[::step]
    cmax = max(cc)
    cmin = min(cc)

    def RX(i):
        return rx0 + i / (len(cc) - 1) * (rx1 - rx0)

    def RY(v):
        return ry0 - (v - cmin) / (cmax - cmin + 1e-9) * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.1"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.1"/>')
    pts = " ".join(f"{RX(i):.1f},{RY(cc[i]):.1f}" for i in range(len(cc)))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#ff6b6b" stroke-width="1.6"/>')
    # mark final level
    parts.append(f'<line x1="{rx0}" y1="{RY(sa_len):.1f}" x2="{rx1}" y2="{RY(sa_len):.1f}" '
                 f'stroke="#06d6a0" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{rx1-2:.1f}" y="{RY(sa_len)-3:.1f}" fill="#06d6a0" font-size="9" '
                 f'text-anchor="end">final {sa_len:.1f}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">iterations (cooling) -> tour length</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
