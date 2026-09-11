"""The Kuramoto model: how oscillators fall into step.

Fireflies flashing in unison, pacemaker cells beating together, applause syncing into rhythm,
power-grid generators locking phase -- all are the same phenomenon: a population of oscillators,
each with its own natural frequency, pulling one another into synchrony through coupling.
Kuramoto (1975) captured it with a single equation for each phase theta_i:

    dtheta_i/dt = omega_i + (K/N) sum_j sin(theta_j - theta_i),

where omega_i is oscillator i's natural frequency and K the coupling strength. The collective
state is measured by the complex order parameter

    r e^{i psi} = (1/N) sum_j e^{i theta_j},

whose magnitude r runs from 0 (phases scattered, incoherent) to 1 (all in phase, fully
synchronized). The startling result is a phase transition: below a critical coupling

    K_c = 2 / (pi g(0)),

with g the distribution of natural frequencies (g(0) its value at the mean), the oscillators
drift independently and r ~ 0; above K_c a synchronized cluster spontaneously forms and r
grows. It is the canonical model of emergent collective order from local interaction.

This module computes the order parameter of a phase set, the Kuramoto derivatives, the
critical coupling for a Lorentzian or Gaussian frequency spread, and a full simulation that
returns the steady-state synchrony versus coupling, and reproduces r=1 for aligned phases,
r=0 for uniform ones, and the incoherent-to-synchronized transition at K_c. Pure stdlib
(seeded LCG, no random module); the collective-dynamics companion to the Van der Pol and
Ising notes.
"""

from __future__ import annotations

import math


class _Rng:
    """Seeded LCG; high bits (low bits of these constants are non-random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def uniform(self, a: float, b: float) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return a + (self.state >> 8) / (1 << 24) * (b - a)


def order_parameter(phases):
    """Magnitude r of the Kuramoto order parameter (1/N) sum e^{i theta}. r=1 fully
    synchronized (all phases equal), r~0 incoherent (phases spread over the circle)."""
    n = len(phases)
    if n == 0:
        return 0.0
    c = sum(math.cos(p) for p in phases) / n
    s = sum(math.sin(p) for p in phases) / n
    return math.sqrt(c * c + s * s)


def mean_phase(phases):
    """Collective phase psi = arg((1/N) sum e^{i theta}) (rad)."""
    c = sum(math.cos(p) for p in phases)
    s = sum(math.sin(p) for p in phases)
    return math.atan2(s, c)


def derivatives(phases, omegas, k):
    """Kuramoto rate for each oscillator: omega_i + (K/N) sum_j sin(theta_j - theta_i)."""
    n = len(phases)
    out = []
    for i in range(n):
        coupling = sum(math.sin(phases[j] - phases[i]) for j in range(n))
        out.append(omegas[i] + (k / n) * coupling)
    return out


def critical_coupling_lorentzian(gamma: float) -> float:
    """Critical coupling K_c = 2 gamma for a Lorentzian spread of natural frequencies with
    half-width gamma (since g(0) = 1/(pi gamma), K_c = 2/(pi g(0)) = 2 gamma)."""
    return 2.0 * gamma


def critical_coupling_gaussian(sigma: float) -> float:
    """Critical coupling for a Gaussian frequency spread of std sigma:
    K_c = 2/(pi g(0)) with g(0) = 1/(sigma sqrt(2 pi)), so K_c = sigma sqrt(8/pi)."""
    return sigma * math.sqrt(8.0 / math.pi)


def simulate(n, k, omega_spread=1.0, dt=0.05, steps=600, seed=1):
    """Simulate N Kuramoto oscillators with uniformly spread natural frequencies in
    [-omega_spread, omega_spread] and coupling K. Returns the time-averaged order parameter r
    over the second half of the run (steady-state synchrony). Uses RK4-free Euler (fine dt)."""
    rng = _Rng(seed)
    omegas = [rng.uniform(-omega_spread, omega_spread) for _ in range(n)]
    phases = [rng.uniform(-math.pi, math.pi) for _ in range(n)]
    rs = []
    for step in range(steps):
        d = derivatives(phases, omegas, k)
        phases = [phases[i] + dt * d[i] for i in range(n)]
        if step >= steps // 2:
            rs.append(order_parameter(phases))
    return sum(rs) / len(rs)
