"""First post-Newtonian (1PN) gravity and the precession of Mercury's perihelion.

Newtonian gravity gives closed ellipses -- a two-body orbit never precesses. The
famous 43 arcseconds/century advance of Mercury's perihelion was the first triumph
of general relativity. To lowest order GR adds a small velocity- and
1/r-dependent correction to the acceleration (the gravitoelectric 1PN term for a
test particle around a mass M):

    a = -GM/r^3 * r_vec
        + GM/(c^2 r^3) * [ (4 GM/r - v^2) r_vec + 4 (r . v) v_vec ]

The exact prediction for the perihelion advance per orbit is

    d(phi) = 6 pi GM / (c^2 a (1 - e^2))

which this module reproduces by direct integration. Units: AU, years, solar
masses, so GM_sun = 4 pi^2 and the speed of light is c = 63239.7 AU/yr.
"""

from __future__ import annotations

import math
from typing import List, Tuple

GM_SUN = 4.0 * math.pi ** 2        # AU^3 / yr^2
C_LIGHT = 63239.7263               # speed of light in AU / yr

ARCSEC_PER_RAD = 180.0 / math.pi * 3600.0


def accel_1pn(state: List[float], GM: float = GM_SUN, c: float = C_LIGHT) -> List[float]:
    """1PN acceleration for state [x, y, vx, vy] (test particle around mass GM)."""
    x, y, vx, vy = state
    r = math.hypot(x, y)
    r2 = r * r
    v2 = vx * vx + vy * vy
    rv = x * vx + y * vy
    # Newtonian part
    pref_n = -GM / (r2 * r)
    ax = pref_n * x
    ay = pref_n * y
    # 1PN correction
    coef = GM / (c * c * r2 * r)
    fr = 4.0 * GM / r - v2
    ax += coef * (fr * x + 4.0 * rv * vx)
    ay += coef * (fr * y + 4.0 * rv * vy)
    return [vx, vy, ax, ay]


def _rk4_step(state, dt, GM, c):
    def f(s):
        return accel_1pn(s, GM, c)
    k1 = f(state)
    k2 = f([state[i] + 0.5 * dt * k1[i] for i in range(4)])
    k3 = f([state[i] + 0.5 * dt * k2[i] for i in range(4)])
    k4 = f([state[i] + dt * k3[i] for i in range(4)])
    return [state[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
            for i in range(4)]


def precession_per_orbit(a: float, e: float, GM: float = GM_SUN,
                         c: float = C_LIGHT, orbits: int = 20,
                         steps_per_orbit: int = 20000) -> float:
    """Integrate a relativistic two-body orbit and measure the mean perihelion
    advance per orbit (radians), by tracking the direction of closest approach."""
    # start at aphelion on +x with vis-viva speed
    r_aph = a * (1.0 + e)
    v_aph = math.sqrt(GM * (2.0 / r_aph - 1.0 / a))
    state = [r_aph, 0.0, 0.0, v_aph]

    period = 2.0 * math.pi * math.sqrt(a ** 3 / GM)
    dt = period / steps_per_orbit

    peri_angles = []
    prev_r = math.hypot(state[0], state[1])
    prev_dr = None
    n_steps = int(orbits * steps_per_orbit)
    for _ in range(n_steps):
        new = _rk4_step(state, dt, GM, c)
        r = math.hypot(new[0], new[1])
        dr = r - prev_r
        # perihelion = local minimum of r: dr goes from - to +
        if prev_dr is not None and prev_dr < 0.0 <= dr:
            # angle of perihelion (use previous point, the actual min)
            peri_angles.append(math.atan2(state[1], state[0]))
        prev_dr = dr
        prev_r = r
        state = new

    if len(peri_angles) < 2:
        return float("nan")
    # unwrap and fit a line: mean advance per orbit
    unwrapped = [peri_angles[0]]
    for ang in peri_angles[1:]:
        prev = unwrapped[-1]
        while ang - prev > math.pi:
            ang -= 2 * math.pi
        while ang - prev < -math.pi:
            ang += 2 * math.pi
        unwrapped.append(ang)
    n = len(unwrapped)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(unwrapped) / n
    num = sum((xs[i] - mx) * (unwrapped[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    return num / den  # radians per orbit


def analytic_precession_per_orbit(a: float, e: float, GM: float = GM_SUN,
                                  c: float = C_LIGHT) -> float:
    """GR closed form: 6 pi GM / (c^2 a (1 - e^2)) radians per orbit."""
    return 6.0 * math.pi * GM / (c * c * a * (1.0 - e * e))


def mercury_precession_arcsec_per_century(c: float = C_LIGHT) -> float:
    """Analytic Mercury perihelion advance in arcsec/century."""
    a, e = 0.387098, 0.205630
    per_orbit = analytic_precession_per_orbit(a, e, GM_SUN, c)
    period = 2.0 * math.pi * math.sqrt(a ** 3 / GM_SUN)  # years
    orbits_per_century = 100.0 / period
    return per_orbit * orbits_per_century * ARCSEC_PER_RAD
