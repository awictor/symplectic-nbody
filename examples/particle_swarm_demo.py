"""Demo: particle swarm optimization on standard benchmarks.

Minimizes the multimodal Rastrigin function, the curved Rosenbrock valley, and the smooth Sphere,
showing the global best plunging to zero, that the swarm beats random search at equal budget, and
that decaying the inertia (explore then exploit) speeds convergence. Draws the swarm's final
positions over the Rastrigin landscape and the convergence curve.

    python examples/particle_swarm_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from particle_swarm import (minimize, random_search, sphere, rastrigin,  # noqa: E402
                            rosenbrock, _Rng)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Particle swarm optimization: a flock of solutions homing on the optimum\n")

    results = []
    for name, fn, bd, opt in [("Sphere", sphere, [(-5.0, 5.0)] * 2, "origin"),
                              ("Rastrigin", rastrigin, [(-5.12, 5.12)] * 2, "origin"),
                              ("Rosenbrock", rosenbrock, [(-2.0, 2.0), (-1.0, 3.0)], "(1,1)")]:
        best, cost = minimize(fn, bd, n_particles=40, n_iter=300, w=0.7, w_final=0.3, seed=1)
        results.append((name, cost, best))
        print(f"  {name:>10} (min at {opt}): cost {cost:.5f} at "
              f"({', '.join(f'{x:.3f}' for x in best)})")

    print("\n  PSO vs random search at equal budget (40x300 = 12000 evaluations):")
    for name, fn, bd in [("Sphere", sphere, [(-5.0, 5.0)] * 2),
                         ("Rastrigin", rastrigin, [(-5.12, 5.12)] * 2)]:
        _, pso = minimize(fn, bd, n_particles=40, n_iter=300, seed=5)
        _, rs = random_search(fn, bd, n_eval=40 * 300, seed=6)
        print(f"    {name:>10}: PSO {pso:.5f}   random {rs:.5f}")

    # convergence of Rastrigin, and the effect of inertia decay
    _, _, hist_decay = minimize(rastrigin, [(-5.12, 5.12)] * 2, n_particles=40, n_iter=200,
                                w=0.9, w_final=0.2, seed=2, track=True)
    _, _, hist_fixed = minimize(rastrigin, [(-5.12, 5.12)] * 2, n_particles=40, n_iter=200,
                                w=0.9, w_final=0.9, seed=2, track=True)
    print("\n  Rastrigin convergence (global best cost) -- decaying inertia converges faster:")
    for it in (0, 10, 30, 60, 120, 199):
        print(f"    iter {it:>3}: decaying-w {hist_decay['history'][it]:7.3f}   "
              f"fixed-w {hist_fixed['history'][it]:7.3f}")

    # final swarm positions on Rastrigin (re-run capturing positions)
    swarm = _final_positions(rastrigin, [(-5.12, 5.12)] * 2, seed=2)

    print("\n  Each particle coasts on inertia, is pulled toward its own best spot (cognitive) and")
    print("  the swarm's best (social); r1,r2 keep it stochastic. High inertia explores, low")
    print("  inertia exploits -- decaying it does both in turn. No gradients, just local rules.")

    _svg(os.path.join(outdir, "particle_swarm.svg"), swarm, hist_decay["history"],
         hist_fixed["history"])
    print(f"\n  wrote {os.path.join(outdir, 'particle_swarm.svg')}")


def _final_positions(fn, bounds, seed):
    """Re-run a short PSO and return the final particle cloud (for the scatter panel)."""
    rng = _Rng(seed)
    n = 40
    dim = len(bounds)
    span = [hi - lo for lo, hi in bounds]
    pos = [[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(n)]
    vel = [[rng.uniform(-0.1 * span[d], 0.1 * span[d]) for d in range(dim)] for _ in range(n)]
    pbest = [list(p) for p in pos]
    pbc = [fn(p) for p in pos]
    g = min(range(n), key=lambda i: pbc[i])
    gbest, gbc = list(pbest[g]), pbc[g]
    for it in range(120):
        w = 0.9 + (0.2 - 0.9) * (it / 119)
        for i in range(n):
            for d in range(dim):
                r1, r2 = rng.uniform(), rng.uniform()
                vel[i][d] = (w * vel[i][d] + 1.5 * r1 * (pbest[i][d] - pos[i][d])
                             + 1.5 * r2 * (gbest[d] - pos[i][d]))
                pos[i][d] += vel[i][d]
                lo, hi = bounds[d]
                pos[i][d] = min(hi, max(lo, pos[i][d]))
            c = fn(pos[i])
            if c < pbc[i]:
                pbest[i], pbc[i] = list(pos[i]), c
                if c < gbc:
                    gbest, gbc = list(pos[i]), c
    return pos


def _svg(path, swarm, hist_decay, hist_fixed, width=760, height=430):
    lo, hi = -5.12, 5.12
    lx0, lx1 = 45, width // 2 - 20
    y0, y1 = height - 50, 72

    def LX(x):
        return lx0 + (x - lo) / (hi - lo) * (lx1 - lx0)

    def LY(y):
        return y0 - (y - lo) / (hi - lo) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Particle swarm: the flock converges on the Rastrigin optimum</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'left: final particle positions clustered at the origin; right: convergence with '
        f'decaying vs fixed inertia</text>',
    ]
    # faint concentric guide rings toward the origin
    for rr in (1.0, 2.5, 4.0):
        parts.append(f'<circle cx="{LX(0):.1f}" cy="{LY(0):.1f}" '
                     f'r="{abs(LX(rr)-LX(0)):.1f}" fill="none" stroke="#21262d" stroke-width="1"/>')
    parts.append(f'<line x1="{lx0}" y1="{y0}" x2="{lx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.1"/>')
    parts.append(f'<line x1="{lx0}" y1="{y0}" x2="{lx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.1"/>')
    # origin marker
    ox, oy = LX(0), LY(0)
    parts.append(f'<line x1="{ox-6:.1f}" y1="{oy:.1f}" x2="{ox+6:.1f}" y2="{oy:.1f}" '
                 f'stroke="#06d6a0" stroke-width="1.6"/>')
    parts.append(f'<line x1="{ox:.1f}" y1="{oy-6:.1f}" x2="{ox:.1f}" y2="{oy+6:.1f}" '
                 f'stroke="#06d6a0" stroke-width="1.6"/>')
    for p in swarm:
        parts.append(f'<circle cx="{LX(p[0]):.1f}" cy="{LY(p[1]):.1f}" r="2.6" '
                     f'fill="#4dabf7" opacity="0.8"/>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">final swarm (green + = optimum)</text>')

    # right: convergence curves (log scale)
    rx0, rx1 = width // 2 + 45, width - 30
    n = len(hist_decay)
    allv = [max(v, 1e-6) for v in hist_decay + hist_fixed]
    logs = [math.log10(v) for v in allv]
    lg, hg = min(logs), max(logs)

    def RX(i):
        return rx0 + i / (n - 1) * (rx1 - rx0)

    def RY(v):
        return y0 - (math.log10(max(v, 1e-6)) - lg) / (hg - lg + 1e-9) * (y0 - y1)

    parts.append(f'<line x1="{rx0}" y1="{y0}" x2="{rx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.1"/>')
    parts.append(f'<line x1="{rx0}" y1="{y0}" x2="{rx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.1"/>')
    dp = " ".join(f"{RX(i):.1f},{RY(hist_decay[i]):.1f}" for i in range(n))
    fp = " ".join(f"{RX(i):.1f},{RY(hist_fixed[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{fp}" fill="none" stroke="#ff922b" stroke-width="1.8"/>')
    parts.append(f'<polyline points="{dp}" fill="none" stroke="#4dabf7" stroke-width="2.2"/>')
    parts.append(f'<text x="{rx0+6}" y="{y1+4}" fill="#4dabf7" font-size="10">decaying inertia</text>')
    parts.append(f'<text x="{rx0+6}" y="{y1+18}" fill="#ff922b" font-size="10">fixed inertia</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">iterations -> global best cost (log)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
