"""The Vicsek model: how self-propelled particles spontaneously align into a flock -- a phase transition in motion.

A flock of starlings, a school of fish, a swarm of bacteria: thousands of individuals moving as one, with
no leader and only local information. The Vicsek model (1995) is the minimal physics of this ACTIVE MATTER
and the founding model of collective motion. N particles move at CONSTANT SPEED on a periodic plane; at
each step every particle sets its heading to the AVERAGE heading of all neighbours within a radius r, plus
a random angular kick of size eta (the NOISE). Nothing else -- no attraction, no repulsion, just noisy
alignment.

That single rule produces a genuine PHASE TRANSITION, controlled by the noise eta. The ORDER PARAMETER

    phi = | (1/N) sum_i (cos theta_i, sin theta_i) |    in [0, 1]

measures how aligned the flock is: phi ~ 1 when everyone moves together, phi ~ 0 when headings are random.
At LOW noise the particles spontaneously break symmetry and flock (phi near 1); at HIGH noise alignment is
destroyed and motion is disordered (phi near 0); between them lies a sharp transition at a critical noise
eta_c (which grows with density). This was one of the first demonstrations that a non-equilibrium system of
self-propelled agents can undergo a symmetry-breaking transition like a magnet -- the model launched the
whole field of active matter.

This module simulates the Vicsek model on a periodic box, measures the polar order parameter, and sweeps
the order-disorder transition in the noise. It uses a seeded RNG. It is validated: at low noise the order
parameter is high (the flock aligns) and at high noise it is low (disorder); order DECREASES monotonically
as noise grows across the transition; the speed of every particle stays constant; higher density raises
the order at fixed noise (more neighbours to align with); a single isolated particle just does a noisy
random walk (order ~ 1 trivially for N=1); and results are reproducible per seed. Pure stdlib; the
active-matter companion to the boids, Kuramoto, and Ising tools."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def _init(n, box, rng):
    """Random positions in [0, box)^2 and random headings in [0, 2pi)."""
    pos = [(rng.u() * box, rng.u() * box) for _ in range(n)]
    theta = [rng.u() * 2 * math.pi for _ in range(n)]
    return pos, theta


def order_parameter(theta):
    """Polar order phi = |mean unit velocity| in [0, 1]. 1 = fully aligned, 0 = disordered."""
    n = len(theta)
    cx = sum(math.cos(t) for t in theta) / n
    cy = sum(math.sin(t) for t in theta) / n
    return math.sqrt(cx * cx + cy * cy)


def _neighbors_mean_angle(pos, theta, i, r, box):
    """Average heading (as an angle) of particles within radius r of particle i, including i itself.

    Averaging is done on the unit circle (sum of cos/sin) to handle the 2pi wrap correctly."""
    xi, yi = pos[i]
    r2 = r * r
    sx = sy = 0.0
    for j in range(len(pos)):
        dx = pos[j][0] - xi
        dy = pos[j][1] - yi
        # minimum-image periodic distance
        dx -= box * round(dx / box)
        dy -= box * round(dy / box)
        if dx * dx + dy * dy <= r2:
            sx += math.cos(theta[j])
            sy += math.sin(theta[j])
    return math.atan2(sy, sx)


def step(pos, theta, r, eta, speed, box, rng):
    """One Vicsek update: align each heading to its neighbourhood mean + noise, then move. In place."""
    n = len(pos)
    new_theta = [0.0] * n
    for i in range(n):
        mean_angle = _neighbors_mean_angle(pos, theta, i, r, box)
        noise = (rng.u() - 0.5) * eta   # uniform in [-eta/2, eta/2]
        new_theta[i] = mean_angle + noise
    for i in range(n):
        theta[i] = new_theta[i]
        x, y = pos[i]
        x = (x + speed * math.cos(theta[i])) % box
        y = (y + speed * math.sin(theta[i])) % box
        pos[i] = (x, y)


def simulate(n, box, r, eta, speed=0.03, steps=200, seed=1, measure_last=50):
    """Run the Vicsek model. Returns the mean order parameter over the last `measure_last` steps
    (after the flock settles) and the final positions/headings."""
    rng = _Rng(seed)
    pos, theta = _init(n, box, rng)
    order_accum = 0.0
    order_count = 0
    for s in range(steps):
        step(pos, theta, r, eta, speed, box, rng)
        if s >= steps - measure_last:
            order_accum += order_parameter(theta)
            order_count += 1
    return {"order": order_accum / order_count if order_count else order_parameter(theta),
            "pos": pos, "theta": theta}


def noise_sweep(n, box, r, etas, speed=0.03, steps=200, seed=1):
    """Order parameter vs noise across the transition. Returns list of (eta, order)."""
    return [(eta, simulate(n, box, r, eta, speed, steps, seed=seed)["order"]) for eta in etas]


def density(n, box):
    """Particle number density rho = N / box^2."""
    return n / (box * box)
