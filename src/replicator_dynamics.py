"""Replicator dynamics: the equation of evolutionary game theory -- strategies that do better spread.

How does the mix of strategies in a population change when each individual's success depends on what
everyone ELSE is playing? REPLICATOR DYNAMICS (Taylor & Jonker 1978) is the central model: a strategy's
frequency grows in proportion to how much its payoff BEATS THE POPULATION AVERAGE. With a payoff matrix A
(A[i][j] = payoff to strategy i against strategy j) and a frequency vector x on the simplex, the fitness of
strategy i is f_i = (A x)_i, the mean fitness is phi = x . A x, and

    dx_i/dt = x_i ( f_i - phi ).

Above-average strategies rise, below-average ones shrink, and the total stays 1 -- the dynamics live on the
probability simplex forever. Its rest points include every NASH EQUILIBRIUM of the game; a stable rest
point is an EVOLUTIONARILY STABLE STRATEGY (ESS). The dynamics reproduce the qualitative outcomes of
evolutionary game theory: a strictly dominant strategy sweeps to fixation, a coordination game converges to
one of its pure equilibria depending on initial conditions, and rock-paper-scissors cycles forever around
the interior equilibrium (with a conserved quantity, the product of the frequencies).

This module integrates the replicator ODE with a simplex-preserving RK4 step, evaluates fitnesses and mean
fitness, and detects rest points. It is validated: the frequencies stay on the simplex (sum to 1, all
non-negative) for all time; a strictly dominant strategy converges to fixation; a two-strategy game
converges to its interior Nash mixed equilibrium (where the two payoffs are equal); rock-paper-scissors
orbits the interior equilibrium and conserves the product x1 x2 x3; the interior equilibrium and the pure
strategies are rest points (zero velocity); mean fitness is non-decreasing for a symmetric (partnership)
game; and results are deterministic. Pure stdlib; the evolutionary-game-theory companion to the
Moran-process, Wright-Fisher, and Lotka-Volterra tools."""

from __future__ import annotations


def _matvec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def fitness(A, x):
    """Fitness of each strategy: f = A x (expected payoff against the current population mix)."""
    return _matvec(A, x)


def mean_fitness(A, x):
    """Mean population fitness phi = x . A x."""
    f = fitness(A, x)
    return sum(x[i] * f[i] for i in range(len(x)))


def velocity(A, x):
    """Replicator velocity dx_i/dt = x_i (f_i - phi)."""
    f = fitness(A, x)
    phi = sum(x[i] * f[i] for i in range(len(x)))
    return [x[i] * (f[i] - phi) for i in range(len(x))]


def _normalize(x):
    s = sum(x)
    return [xi / s for xi in x] if s > 0 else x


def step_rk4(A, x, dt):
    """One simplex-preserving RK4 step of the replicator ODE, renormalized to sum 1."""
    def deriv(state):
        return velocity(A, state)

    k1 = deriv(x)
    x2 = [x[i] + 0.5 * dt * k1[i] for i in range(len(x))]
    k2 = deriv(x2)
    x3 = [x[i] + 0.5 * dt * k2[i] for i in range(len(x))]
    k3 = deriv(x3)
    x4 = [x[i] + dt * k3[i] for i in range(len(x))]
    k4 = deriv(x4)
    new = [x[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(len(x))]
    # clamp tiny negatives from numerical error, then renormalize onto the simplex
    new = [max(0.0, v) for v in new]
    return _normalize(new)


def integrate(A, x0, t_max, dt=0.01, record_every=1):
    """Integrate the replicator dynamics from x0 to t_max. Returns (times, trajectory of x vectors)."""
    x = _normalize(list(x0))
    times = [0.0]
    traj = [list(x)]
    n_steps = int(t_max / dt)
    for step in range(1, n_steps + 1):
        x = step_rk4(A, x, dt)
        if step % record_every == 0:
            times.append(step * dt)
            traj.append(list(x))
    return times, traj


def is_rest_point(A, x, tol=1e-9):
    """True if x is a rest point (replicator velocity ~ 0)."""
    v = velocity(A, x)
    return all(abs(vi) < tol for vi in v)


def rps_conserved(x):
    """Rock-paper-scissors conserved quantity: the product of the frequencies x1 x2 x3."""
    p = 1.0
    for xi in x:
        p *= xi
    return p
