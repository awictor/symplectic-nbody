"""Gravitational-wave inspiral: the LIGO chirp from first principles.

A tight binary radiates energy as gravitational waves. The orbit shrinks, the
orbital frequency sweeps upward, and the two bodies spiral together -- the
"chirp" LIGO heard from merging black holes. To leading order the effect is the
2.5PN radiation-reaction force (Burke-Thorne), which we add to Newtonian gravity.

Peters (1964) gives the exact leading-order predictions this module reproduces:

  * energy loss rate  dE/dt = -(32/5) G^4 mu^2 M^3 / (c^5 a^5)   (circular)
  * coalescence time  t_c = (5/256) c^5 a0^4 / (G^3 mu M^2)
  * frequency chirp   f_gw(t) ~ (t_c - t)^(-3/8)

Units: G = 1, masses in solar masses, length in AU, time in years. The speed of
light is c = 63239.7 AU/yr, so at astrophysical separations the effect is tiny;
the demo shrinks c to bring an inspiral into view, exactly as with the perihelion
precession. Pure stdlib.
"""

from __future__ import annotations

import math
from typing import List, Tuple

G = 1.0
C_LIGHT = 63239.7263  # AU / yr


def relative_accel(r: List[float], v: List[float], m1: float, m2: float,
                   c: float = C_LIGHT) -> List[float]:
    """Acceleration of the relative separation r = r2 - r1, including the 2.5PN
    radiation-reaction term. Returns a 3-vector d^2 r / dt^2."""
    M = m1 + m2
    x, y, z = r
    dist = math.sqrt(x * x + y * y + z * z)
    d3 = dist ** 3
    # Newtonian relative acceleration
    a = [-G * M * x / d3, -G * M * y / d3, -G * M * z / d3]

    # 2.5PN radiation reaction (relative motion), leading dissipative term:
    #   a_RR = (8/5) G^2 M mu / (c^5 r^3) * [ (18 v^2 - 6 G M/r ... ) ... ]
    # We use the standard circular-compatible Burke-Thorne relative form:
    #   a_RR = -(8/5) (G^2 M mu)/(c^5 r^3) * ( (3 v^2 + 17/3 GM/r) (r_hat) ... )
    # For robustness we use the widely-quoted vector form (Damour):
    mu = m1 * m2 / M
    rv = x * v[0] + y * v[1] + z * v[2]
    v2 = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
    pref = (8.0 / 5.0) * G * G * M * mu / (c ** 5 * d3)
    # a_RR_i = pref * [ (rv/r^2)(3 v^2 + (17/3) GM/r) r_i  -  (v^2 + 3 GM/r) v_i ]
    A = (rv / (dist * dist)) * (3.0 * v2 + (17.0 / 3.0) * G * M / dist)
    B = (v2 + 3.0 * G * M / dist)
    a[0] += pref * (A * x - B * v[0])
    a[1] += pref * (A * y - B * v[1])
    a[2] += pref * (A * z - B * v[2])
    return a


def _rk4_rel(r, v, m1, m2, c, dt):
    def f(state):
        rr = state[:3]
        vv = state[3:]
        acc = relative_accel(rr, vv, m1, m2, c)
        return [vv[0], vv[1], vv[2], acc[0], acc[1], acc[2]]
    s = list(r) + list(v)
    k1 = f(s)
    k2 = f([s[i] + 0.5 * dt * k1[i] for i in range(6)])
    k3 = f([s[i] + 0.5 * dt * k2[i] for i in range(6)])
    k4 = f([s[i] + dt * k3[i] for i in range(6)])
    s = [s[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(6)]
    return s[:3], s[3:]


def coalescence_time(a0: float, e0: float, m1: float, m2: float,
                     c: float = C_LIGHT) -> float:
    """Peters (1964) merger time for a circular binary (e0=0). For e0>0 it is
    shorter; we return the circular value scaled by the standard (1-e^2)^{7/2}
    enhancement as a first approximation."""
    M = m1 + m2
    mu = m1 * m2 / M
    t_circ = (5.0 / 256.0) * c ** 5 * a0 ** 4 / (G ** 3 * mu * M * M)
    return t_circ * (1.0 - e0 * e0) ** 3.5


def inspiral(a0: float, m1: float, m2: float, c: float,
             dt: float, n_steps: int, sample_every: int = 1
             ) -> Tuple[List[float], List[float], List[float]]:
    """Integrate a circular binary of initial separation a0 with radiation
    reaction. Returns (times, separations, orbital_frequencies)."""
    M = m1 + m2
    # start on a circular orbit in the xy-plane
    r = [a0, 0.0, 0.0]
    v_circ = math.sqrt(G * M / a0)
    v = [0.0, v_circ, 0.0]
    ts, seps, freqs = [], [], []
    t = 0.0
    for s in range(n_steps):
        r, v = _rk4_rel(r, v, m1, m2, c, dt)
        t += dt
        dist = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        if dist < 1e-4 * a0:  # effectively merged
            break
        if s % sample_every == 0:
            speed = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            f_orb = speed / (2.0 * math.pi * dist)  # instantaneous orbital freq
            ts.append(t); seps.append(dist); freqs.append(f_orb)
    return ts, seps, freqs
