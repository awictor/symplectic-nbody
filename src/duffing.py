"""The Duffing oscillator: a spring that bends the rules.

A normal spring pulls back proportionally to how far you stretch it (F = -kx). Stiffen or
soften it at large amplitude and you get the Duffing oscillator, the simplest nonlinear
spring:

    x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t),

with delta the damping, alpha the linear stiffness, beta the cubic (nonlinear) stiffness,
gamma and omega the drive. The cubic term does remarkable things. Its unforced potential is

    V(x) = (1/2) alpha x^2 + (1/4) beta x^4,

which for alpha < 0, beta > 0 is a DOUBLE WELL -- two stable rest points separated by a hill,
the model for a buckled beam or a bead on a rotating hoop. Driven, the same system shows the
hallmark nonlinear resonance: the response curve bends over, so amplitude is a multi-valued
function of drive frequency and the system JUMPS between a high and a low branch as you sweep
frequency (hysteresis). Push it harder and the forced Duffing becomes chaotic, one of the
canonical routes to a strange attractor.

This module integrates the oscillator (RK4), gives the double-well potential and its minima,
the (softening/hardening/double-well) regime, the backbone curve of the amplitude-frequency
bend, and a driven trajectory, and reproduces the +/-sqrt(-alpha/beta) well minima and the
frequency shift of the resonance peak with amplitude. Dimensionless. Pure stdlib; the
nonlinear-resonance companion to the Van der Pol and pendulum notes.
"""

from __future__ import annotations

import math


def potential(x, alpha, beta):
    """Unforced Duffing potential V(x) = (1/2) alpha x^2 + (1/4) beta x^4. A single well for
    alpha>0, a double well for alpha<0, beta>0."""
    return 0.5 * alpha * x * x + 0.25 * beta * x ** 4


def well_minima(alpha, beta):
    """Locations of the potential minima. For a double well (alpha<0, beta>0):
    x = +/- sqrt(-alpha/beta). Otherwise the single minimum at x=0."""
    if alpha < 0 and beta > 0:
        x = math.sqrt(-alpha / beta)
        return [-x, x]
    return [0.0]


def is_double_well(alpha, beta) -> bool:
    """True if the unforced potential is a double well (two stable states): alpha<0, beta>0."""
    return alpha < 0.0 and beta > 0.0


def regime(alpha, beta) -> str:
    """Classify the spring: 'hardening' (beta>0, alpha>0 -- stiffens with amplitude),
    'softening' (beta<0), 'double-well' (alpha<0, beta>0), or 'linear' (beta=0)."""
    if beta == 0.0:
        return "linear"
    if alpha < 0.0 and beta > 0.0:
        return "double-well"
    return "hardening" if beta > 0.0 else "softening"


def derivatives(state, t, delta, alpha, beta, gamma, omega):
    """Vector field (dx, dv) for state (x, v): dx = v,
    dv = -delta v - alpha x - beta x^3 + gamma cos(omega t)."""
    x, v = state
    return (v, -delta * v - alpha * x - beta * x ** 3 + gamma * math.cos(omega * t))


def rk4_step(state, t, dt, delta, alpha, beta, gamma, omega):
    """One RK4 step of the (possibly driven) Duffing oscillator."""
    def add(s, k, f):
        return (s[0] + f * k[0], s[1] + f * k[1])
    k1 = derivatives(state, t, delta, alpha, beta, gamma, omega)
    k2 = derivatives(add(state, k1, dt / 2), t + dt / 2, delta, alpha, beta, gamma, omega)
    k3 = derivatives(add(state, k2, dt / 2), t + dt / 2, delta, alpha, beta, gamma, omega)
    k4 = derivatives(add(state, k3, dt), t + dt, delta, alpha, beta, gamma, omega)
    return (state[0] + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            state[1] + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


def trajectory(state0, dt, n, delta=0.0, alpha=1.0, beta=0.0, gamma=0.0, omega=1.0):
    """Integrate n RK4 steps; return the list of (x, v) states (length n+1)."""
    pts = [tuple(state0)]
    s = tuple(state0)
    t = 0.0
    for _ in range(n):
        s = rk4_step(s, t, dt, delta, alpha, beta, gamma, omega)
        t += dt
        pts.append(s)
    return pts


def backbone_frequency(amplitude, alpha, beta) -> float:
    """Backbone curve: the amplitude-dependent natural frequency of the undamped, unforced
    oscillator, omega ~ sqrt(alpha + (3/4) beta A^2). For a hardening spring (beta>0) the
    resonance peak shifts UP with amplitude -- the bend that causes the jump/hysteresis."""
    val = alpha + 0.75 * beta * amplitude * amplitude
    return math.sqrt(val) if val > 0 else 0.0
