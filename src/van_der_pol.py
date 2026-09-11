"""The Van der Pol oscillator: a rhythm that regulates itself.

A pendulum loses energy and dies; a Van der Pol oscillator pumps itself. Balthasar van der Pol
built it in the 1920s from a vacuum-tube circuit whose damping changes sign with amplitude:

    x'' - mu (1 - x^2) x' + x = 0.

When the swing is small (|x| < 1) the damping term is negative -- the circuit feeds energy in
and the oscillation grows. When it is large (|x| > 1) the damping is positive and drains
energy. The two effects balance at a unique amplitude, so from almost any start the system
settles onto the same closed loop in phase space -- a LIMIT CYCLE. Unlike a linear oscillator
(whose amplitude depends on how you started it), a limit cycle forgets its initial conditions:
it is the mathematical model of a self-sustaining rhythm -- heartbeats, neuron firing, the
circadian clock, a bowed violin string.

The parameter mu sets the character. Small mu gives nearly sinusoidal oscillation at angular
frequency ~1. Large mu gives relaxation oscillation: long slow charges punctuated by fast
jumps, with period growing roughly as (3 - 2 ln 2) mu ~ 1.614 mu. Push it with an external
drive and it can entrain (phase-lock) or go chaotic -- the forced Van der Pol was one of the
first systems where chaos was seen.

This module integrates the oscillator (RK4), gives the energy-like Lienard quantity, the
amplitude of the settled limit cycle, and the large-mu relaxation period, and reproduces the
limit-cycle amplitude ~2 and the convergence of different starts onto the same cycle. SI-free
(dimensionless). Pure stdlib; the self-oscillation companion to the Lorenz and pendulum notes.
"""

from __future__ import annotations

import math


def derivatives(state, mu: float):
    """Van der Pol vector field (dx, dv) for state (x, v), where v = x':
    dx = v, dv = mu (1 - x^2) v - x."""
    x, v = state
    return (v, mu * (1.0 - x * x) * v - x)


def rk4_step(state, dt, mu):
    """One RK4 step of the Van der Pol oscillator."""
    def add(s, k, f):
        return (s[0] + f * k[0], s[1] + f * k[1])
    k1 = derivatives(state, mu)
    k2 = derivatives(add(state, k1, dt / 2), mu)
    k3 = derivatives(add(state, k2, dt / 2), mu)
    k4 = derivatives(add(state, k3, dt), mu)
    return (state[0] + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            state[1] + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


def trajectory(state0, dt, n, mu):
    """Integrate n RK4 steps; return the list of (x, v) states (length n+1)."""
    pts = [tuple(state0)]
    s = tuple(state0)
    for _ in range(n):
        s = rk4_step(s, dt, mu)
        pts.append(s)
    return pts


def limit_cycle_amplitude(mu, dt=0.01, settle=4000, sample=2000):
    """Peak |x| of the settled limit cycle: integrate past the transient, then take the max
    |x| over a sampling window. ~2 for all mu (the classic Van der Pol amplitude)."""
    s = (0.1, 0.0)
    for _ in range(settle):
        s = rk4_step(s, dt, mu)
    peak = 0.0
    for _ in range(sample):
        s = rk4_step(s, dt, mu)
        peak = max(peak, abs(s[0]))
    return peak


def relaxation_period(mu) -> float:
    """Large-mu relaxation-oscillation period, T ~ (3 - 2 ln 2) mu ~ 1.614 mu. Valid for
    mu >> 1; the period grows linearly with mu as slow charges dominate."""
    return (3.0 - 2.0 * math.log(2.0)) * mu


def is_self_sustaining(mu) -> bool:
    """True if the oscillator sustains itself (a stable limit cycle exists): mu > 0. At mu = 0
    it is a conservative linear oscillator; mu < 0 is damped to rest."""
    return mu > 0.0


def measured_period(mu, dt=0.005, settle=6000, span=20000) -> float:
    """Period (time units) of the settled oscillation, from successive upward zero-crossings
    of x after the transient."""
    s = (0.1, 0.0)
    for _ in range(settle):
        s = rk4_step(s, dt, mu)
    crossings = []
    prev = s
    t = 0.0
    for _ in range(span):
        s = rk4_step(s, dt, mu)
        t += dt
        if prev[0] < 0.0 <= s[0]:      # upward zero-crossing
            crossings.append(t)
        prev = s
    if len(crossings) < 2:
        return float("nan")
    gaps = [crossings[i + 1] - crossings[i] for i in range(len(crossings) - 1)]
    return sum(gaps) / len(gaps)
