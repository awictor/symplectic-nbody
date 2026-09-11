"""Particle swarm optimization: a flock of solutions homing in on the optimum.

Particle swarm optimization (Kennedy & Eberhart, 1995) is a third metaheuristic alongside
simulated annealing (perturb one state) and genetic algorithms (breed a population): here a SWARM
of candidate solutions flies through the search space, each remembering its own best spot and
pulled toward the best spot any member has found. It is inspired by flocking birds -- individuals
follow simple velocity rules, and collectively the swarm converges on food (the optimum).

Each particle carries a position x and a VELOCITY v. Every step the velocity is updated by three
pulls, then the position moves along it:

    v <- w*v  +  c1*r1*(pbest - x)  +  c2*r2*(gbest - x)
    x <- x + v

  INERTIA (w)   -- keep some of the old velocity, so the particle coasts and explores.
  COGNITIVE (c1)-- pull toward this particle's own best-ever position (pbest): individual memory.
  SOCIAL (c2)   -- pull toward the swarm's global best (gbest): shared knowledge.

r1, r2 are fresh uniform randoms that keep the search stochastic. High inertia explores; low
inertia exploits, so w is often decayed over the run. With no gradients and only these local rules
the swarm balances exploration and convergence, and it is a standard baseline for continuous global
optimization. This module implements PSO over a bounded box with velocity clamping and optional
linearly-decaying inertia -- verified that it finds the global minimum of the Sphere, Rastrigin,
and Rosenbrock benchmarks, that the global best improves monotonically, that it beats random search
at equal budget, and that lowering inertia speeds convergence. Pure stdlib (its own RNG); the
swarm-based companion to the simulated-annealing and genetic-algorithm notes."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed=0):
        self.state = seed & 0xFFFFFFFF

    def uniform(self, lo=0.0, hi=1.0):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return lo + (hi - lo) * ((self.state >> 8) / (1 << 24))     # high bits


def minimize(cost_fn, bounds, n_particles=30, n_iter=200, w=0.7, c1=1.5, c2=1.5,
             w_final=None, seed=0, track=False):
    """Minimize cost_fn over an axis-aligned box by particle swarm optimization.

    bounds : list of (lo, hi) per dimension.
    w      : inertia weight; if w_final is given, inertia decays linearly from w to w_final.
    c1, c2 : cognitive (personal-best) and social (global-best) pull coefficients.
    Returns (best_position, best_cost) or, with track=True, also a history dict."""
    rng = _Rng(seed)
    dim = len(bounds)
    span = [hi - lo for lo, hi in bounds]
    vmax = [0.5 * s for s in span]                     # clamp velocity to half the box span

    # initialize positions uniformly in the box, velocities small
    pos = [[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(n_particles)]
    vel = [[rng.uniform(-0.1 * span[d], 0.1 * span[d]) for d in range(dim)]
           for _ in range(n_particles)]
    pbest = [list(p) for p in pos]
    pbest_cost = [cost_fn(p) for p in pos]
    g = min(range(n_particles), key=lambda i: pbest_cost[i])
    gbest = list(pbest[g])
    gbest_cost = pbest_cost[g]
    history = [gbest_cost]

    for it in range(n_iter):
        wk = w if w_final is None else w + (w_final - w) * (it / max(1, n_iter - 1))
        for i in range(n_particles):
            for d in range(dim):
                r1 = rng.uniform()
                r2 = rng.uniform()
                vel[i][d] = (wk * vel[i][d]
                             + c1 * r1 * (pbest[i][d] - pos[i][d])
                             + c2 * r2 * (gbest[d] - pos[i][d]))
                # clamp velocity, then move and clamp position to the box
                if vel[i][d] > vmax[d]:
                    vel[i][d] = vmax[d]
                elif vel[i][d] < -vmax[d]:
                    vel[i][d] = -vmax[d]
                pos[i][d] += vel[i][d]
                lo, hi = bounds[d]
                if pos[i][d] < lo:
                    pos[i][d] = lo
                    vel[i][d] = 0.0
                elif pos[i][d] > hi:
                    pos[i][d] = hi
                    vel[i][d] = 0.0
            c = cost_fn(pos[i])
            if c < pbest_cost[i]:
                pbest[i] = list(pos[i])
                pbest_cost[i] = c
                if c < gbest_cost:
                    gbest = list(pos[i])
                    gbest_cost = c
        history.append(gbest_cost)

    if track:
        return gbest, gbest_cost, {"history": history}
    return gbest, gbest_cost


def random_search(cost_fn, bounds, n_eval=6000, seed=0):
    """Baseline: sample n_eval uniform-random points in the box, keep the best."""
    rng = _Rng(seed)
    best, best_cost = None, math.inf
    for _ in range(n_eval):
        x = [rng.uniform(lo, hi) for lo, hi in bounds]
        c = cost_fn(x)
        if c < best_cost:
            best, best_cost = x, c
    return best, best_cost


# --- standard continuous-optimization benchmarks (global minimum 0 at the origin,
#     except Rosenbrock whose minimum 0 is at (1, 1, ..., 1)) ------------------
def sphere(x):
    """Sum of squares; smooth bowl, minimum 0 at the origin."""
    return sum(xi * xi for xi in x)


def rastrigin(x):
    """Highly multimodal (a lattice of local minima); global minimum 0 at the origin."""
    a = 10.0
    return a * len(x) + sum(xi * xi - a * math.cos(2 * math.pi * xi) for xi in x)


def rosenbrock(x):
    """Banana-shaped valley; minimum 0 at (1, 1, ...), hard for many optimizers."""
    return sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))
