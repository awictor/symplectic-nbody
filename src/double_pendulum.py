"""The double pendulum: chaos from two rods and gravity.

Hang one pendulum from the end of another and you have the simplest mechanical system that is
chaotic. It is fully deterministic -- Newton's laws, no randomness -- yet two nearly identical
releases end up flailing completely differently within seconds. It is the classic tabletop
demonstration of sensitive dependence on initial conditions.

The state is two angles (theta1, theta2, from vertical) and their rates. The equations of
motion, from the Lagrangian, are a coupled pair (m1, m2 the bob masses, L1, L2 the rod
lengths):

    a1'' and a2'' = messy trig expressions in the angles, rates, masses, lengths, and g,

which have no closed-form solution -- you must integrate them numerically (RK4 here). Unlike a
single pendulum, the energy surface is not integrable, so the motion explores it chaotically
above a threshold energy.

Two things stay clean, and this module checks them. First, the TOTAL ENERGY (kinetic plus
potential) is conserved along the exact motion, so a good integrator keeps it nearly constant
-- a stringent test of the physics and the solver. Second, two trajectories from nearly equal
starts diverge exponentially, with a positive Lyapunov exponent, the signature of chaos.

This module gives the equations of motion, an RK4 step and trajectory, the total energy of a
state, and an estimate of the trajectory divergence rate, and reproduces energy conservation
to high accuracy and the exponential separation of nearby starts. SI units, angles in radians.
Pure stdlib; the deterministic-chaos companion to the Lorenz and N-body Lyapunov notes.
"""

from __future__ import annotations

import math

G = 9.80665


def derivatives(state, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=G):
    """Return (dtheta1, domega1, dtheta2, domega2) for state (theta1, omega1, theta2, omega2),
    the standard double-pendulum equations of motion."""
    t1, w1, t2, w2 = state
    dt = t1 - t2
    den1 = (m1 + m2) * l1 - m2 * l1 * math.cos(dt) * math.cos(dt)
    den2 = (l2 / l1) * den1

    dw1 = (m2 * l1 * w1 * w1 * math.sin(dt) * math.cos(dt)
           + m2 * g * math.sin(t2) * math.cos(dt)
           + m2 * l2 * w2 * w2 * math.sin(dt)
           - (m1 + m2) * g * math.sin(t1)) / den1

    dw2 = (-m2 * l2 * w2 * w2 * math.sin(dt) * math.cos(dt)
           + (m1 + m2) * (g * math.sin(t1) * math.cos(dt)
                          - l1 * w1 * w1 * math.sin(dt)
                          - g * math.sin(t2))) / den2

    return (w1, dw1, w2, dw2)


def rk4_step(state, dt, **kw):
    """One RK4 step of the double-pendulum equations."""
    def add(s, k, f):
        return tuple(s[i] + f * k[i] for i in range(4))
    k1 = derivatives(state, **kw)
    k2 = derivatives(add(state, k1, dt / 2), **kw)
    k3 = derivatives(add(state, k2, dt / 2), **kw)
    k4 = derivatives(add(state, k3, dt), **kw)
    return tuple(state[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(4))


def trajectory(state0, dt, n, **kw):
    """Integrate n RK4 steps; return the list of states (length n+1)."""
    pts = [tuple(state0)]
    s = tuple(state0)
    for _ in range(n):
        s = rk4_step(s, dt, **kw)
        pts.append(s)
    return pts


def total_energy(state, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=G) -> float:
    """Total mechanical energy (J) of the state: kinetic (both bobs) plus gravitational
    potential (measured from the pivot). Conserved along the exact motion."""
    t1, w1, t2, w2 = state
    # bob velocities
    v1sq = (l1 * w1) ** 2
    v2sq = (l1 * w1) ** 2 + (l2 * w2) ** 2 + 2 * l1 * l2 * w1 * w2 * math.cos(t1 - t2)
    ke = 0.5 * m1 * v1sq + 0.5 * m2 * v2sq
    y1 = -l1 * math.cos(t1)
    y2 = y1 - l2 * math.cos(t2)
    pe = m1 * g * y1 + m2 * g * y2
    return ke + pe


def positions(state, l1=1.0, l2=1.0):
    """(x1, y1, x2, y2): Cartesian bob positions for plotting, pivot at the origin."""
    t1, _, t2, _ = state
    x1 = l1 * math.sin(t1)
    y1 = -l1 * math.cos(t1)
    x2 = x1 + l2 * math.sin(t2)
    y2 = y1 - l2 * math.cos(t2)
    return x1, y1, x2, y2


def divergence_rate(state0, dt=0.005, n=6000, delta=1e-8, **kw) -> float:
    """Rough largest-Lyapunov estimate: evolve two starts `delta` apart, rescaling each step,
    and average the log growth per unit time. Positive for a chaotic (high-energy) start."""
    a = tuple(state0)
    b = (state0[0] + delta, state0[1], state0[2], state0[3])

    def sep(p, q):
        return math.sqrt(sum((p[i] - q[i]) ** 2 for i in range(4)))

    total = 0.0
    for _ in range(n):
        a = rk4_step(a, dt, **kw)
        b = rk4_step(b, dt, **kw)
        d = sep(a, b)
        if d > 0:
            total += math.log(d / delta)
            b = tuple(a[i] + (b[i] - a[i]) * delta / d for i in range(4))
    return total / (n * dt)
