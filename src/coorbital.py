"""Coorbital motion: tadpole and horseshoe orbits in the CR3BP.

A test particle sharing a planet's orbit (same semi-major axis, so nearly the
same period) does not simply sit still in the rotating frame -- it slowly
librates in the co-rotating potential:

  * a TADPOLE orbit loops around a single triangular Lagrange point (L4 or L5),
    tracing a tadpole-shaped path -- e.g. Jupiter's Trojan asteroids;
  * a HORSESHOE orbit swings all the way around L3, turning back before it
    reaches the planet at each end, so it encloses BOTH L4 and L5 -- e.g.
    Saturn's coorbital moons Janus and Epimetheus, and Earth's companion 3753
    Cruithne.

The distinction is the angular range the particle covers relative to the
secondary: a tadpole stays on one side (range well under 180 deg); a horseshoe
sweeps across the far side (range > 180 deg, approaching 360).

Integrated in the CR3BP rotating frame (reused from cr3bp.py). Pure stdlib.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from cr3bp import CR3BP


def _rk4(f, s, dt):
    k1 = f(s)
    k2 = f([s[i] + 0.5 * dt * k1[i] for i in range(4)])
    k3 = f([s[i] + 0.5 * dt * k2[i] for i in range(4)])
    k4 = f([s[i] + dt * k3[i] for i in range(4)])
    return [s[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(4)]


def _angle_about_origin(x: float, y: float) -> float:
    """Polar angle of (x, y) in [0, 2*pi). The secondary sits near angle 0."""
    a = math.atan2(y, x)
    return a + 2.0 * math.pi if a < 0 else a


def coorbital_trajectory(model: CR3BP, x0: float, y0: float,
                         vx0: float = 0.0, vy0: float = 0.0,
                         dt: float = 0.005, steps: int = 200000,
                         sample_every: int = 50):
    """Integrate a coorbital test particle in the rotating frame. Returns
    (xs, ys, angles) where angles are the particle's polar angle each sample."""
    s = [x0, y0, vx0, vy0]
    xs, ys, angs = [x0], [y0], [_angle_about_origin(x0, y0)]
    for i in range(steps):
        s = _rk4(model.accel, s, dt)
        if abs(s[0]) > 5 or abs(s[1]) > 5:      # escaped the coorbital region
            break
        if i % sample_every == 0:
            xs.append(s[0]); ys.append(s[1])
            angs.append(_angle_about_origin(s[0], s[1]))
    return xs, ys, angs


def angular_range(angles: List[float]) -> float:
    """Peak-to-peak angular excursion (degrees) of the particle about the primary,
    measured relative to the secondary at angle 0. Robustly handles the wrap by
    unwrapping the sequence first."""
    unwrapped = [angles[0]]
    for a in angles[1:]:
        prev = unwrapped[-1]
        while a - prev > math.pi:
            a -= 2 * math.pi
        while a - prev < -math.pi:
            a += 2 * math.pi
        unwrapped.append(a)
    return math.degrees(max(unwrapped) - min(unwrapped))


def start_near_L4(model: CR3BP, offset: float = 0.0):
    """Initial (x, y) a small radial offset from the L4 triangular point.
    In the rotating frame L4 is an equilibrium, so a small nudge gives a bounded
    tadpole libration. Keep |offset| small (<~0.01) or the orbit escapes."""
    (x4, y4) = model.lagrange_points()["L4"]
    r = math.hypot(x4, y4)
    ux, uy = x4 / r, y4 / r
    return x4 + offset * ux, y4 + offset * uy


def start_on_corotation(angle_deg: float):
    """Initial (x, y) on the corotation circle (unit radius) at a given polar
    angle, at rest in the rotating frame. The secondary sits at angle 0. Start
    near 180 deg (opposite the secondary, by L3) to launch a horseshoe; start
    near 60 or 300 deg (by L4/L5) for a tadpole."""
    a = math.radians(angle_deg)
    return math.cos(a), math.sin(a)


def classify(angles: List[float]) -> str:
    """'tadpole' if the particle stays on one side of the primary-secondary line
    (angular range < 180 deg), else 'horseshoe'."""
    return "tadpole" if angular_range(angles) < 180.0 else "horseshoe"
