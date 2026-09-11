"""The Lorenz attractor: the butterfly that killed long-range prediction.

In 1963 Edward Lorenz stripped atmospheric convection down to three coupled equations and
found that even this toy weather never repeats and cannot be predicted for long:

    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = x y - beta z.

At the classic parameters sigma = 10, beta = 8/3, rho = 28 the trajectory winds forever around
two spiral lobes, jumping between them unpredictably -- the butterfly-shaped Lorenz attractor,
the first and most famous strange attractor of a continuous flow. Two trajectories starting a
millionth apart diverge until they are on opposite wings: the "butterfly effect," sensitive
dependence on initial conditions, which is why weather is unforecastable beyond ~two weeks.

The flow is dissipative -- phase-space volume shrinks at the constant rate

    div F = -(sigma + 1 + beta),

so all trajectories collapse onto the zero-volume (fractal, dimension ~2.06) attractor while
never settling. Below rho = 1 the origin is the only, stable equilibrium; above it two new
fixed points appear at (+/-sqrt(beta(rho-1)), same, rho-1), and past rho ~ 24.74 they too go
unstable and chaos reigns. The largest Lyapunov exponent is ~0.906, positive: prediction error
grows ten-fold every ~2.5 time units.

This module integrates the system (RK4), gives the volume-contraction rate and the nontrivial
fixed points, and estimates the largest Lyapunov exponent from two nearby trajectories, and
reproduces the -(sigma+1+beta) contraction and the positive Lyapunov exponent of the classic
butterfly. Pure stdlib; the continuous-chaos companion to the Henon and N-body Lyapunov notes.
"""

from __future__ import annotations

import math

SIGMA, BETA, RHO = 10.0, 8.0 / 3.0, 28.0     # classic Lorenz parameters


def derivatives(state, sigma=SIGMA, beta=BETA, rho=RHO):
    """The Lorenz vector field (dx, dy, dz) at a phase-space point (x, y, z)."""
    x, y, z = state
    return (sigma * (y - x), x * (rho - z) - y, x * y - beta * z)


def rk4_step(state, dt, sigma=SIGMA, beta=BETA, rho=RHO):
    """One classical Runge-Kutta 4 step of the Lorenz flow."""
    def add(s, k, f):
        return (s[0] + f * k[0], s[1] + f * k[1], s[2] + f * k[2])
    k1 = derivatives(state, sigma, beta, rho)
    k2 = derivatives(add(state, k1, dt / 2), sigma, beta, rho)
    k3 = derivatives(add(state, k2, dt / 2), sigma, beta, rho)
    k4 = derivatives(add(state, k3, dt), sigma, beta, rho)
    return (state[0] + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            state[1] + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            state[2] + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]))


def trajectory(state0, dt, n, sigma=SIGMA, beta=BETA, rho=RHO):
    """Integrate n RK4 steps and return the list of (x, y, z) states (length n+1)."""
    pts = [tuple(state0)]
    s = tuple(state0)
    for _ in range(n):
        s = rk4_step(s, dt, sigma, beta, rho)
        pts.append(s)
    return pts


def volume_contraction_rate(sigma=SIGMA, beta=BETA) -> float:
    """Divergence of the flow, div F = -(sigma + 1 + beta): the constant rate at which any
    phase-space volume shrinks. Negative (dissipative) -> collapse onto the attractor."""
    return -(sigma + 1.0 + beta)


def fixed_points(sigma=SIGMA, beta=BETA, rho=RHO):
    """Equilibria of the Lorenz system. The origin always; for rho > 1 also the two
    convection points C+/- = (+/-sqrt(beta(rho-1)), same, rho-1)."""
    fps = [(0.0, 0.0, 0.0)]
    if rho > 1.0:
        c = math.sqrt(beta * (rho - 1.0))
        fps.append((c, c, rho - 1.0))
        fps.append((-c, -c, rho - 1.0))
    return fps


def largest_lyapunov(state0=(1.0, 1.0, 1.0), dt=0.01, n=20000,
                     sigma=SIGMA, beta=BETA, rho=RHO, d0=1e-9) -> float:
    """Largest Lyapunov exponent from two trajectories started d0 apart, rescaling the
    separation each step and averaging the log growth. ~0.906 for the classic butterfly;
    positive => chaos."""
    a = tuple(state0)
    b = (state0[0] + d0, state0[1], state0[2])
    # settle onto the attractor first
    for _ in range(2000):
        a = rk4_step(a, dt, sigma, beta, rho)
        b = rk4_step(b, dt, sigma, beta, rho)
    # renormalize the separation to d0
    def sep(p, q):
        return math.sqrt(sum((p[i] - q[i]) ** 2 for i in range(3)))
    d = sep(a, b)
    b = tuple(a[i] + (b[i] - a[i]) * d0 / d for i in range(3))
    total = 0.0
    for _ in range(n):
        a = rk4_step(a, dt, sigma, beta, rho)
        b = rk4_step(b, dt, sigma, beta, rho)
        d = sep(a, b)
        total += math.log(d / d0)
        b = tuple(a[i] + (b[i] - a[i]) * d0 / d for i in range(3))
    return total / (n * dt)
