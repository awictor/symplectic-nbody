"""Numerical integrators for N-body gravitational dynamics.

The point of this module is to make one property observable: *symplectic*
integrators (leapfrog / velocity-Verlet, forest-ruth) conserve a system's total
energy over exponentially long times, while a naive high-order method (RK4)
does not — its energy drifts secularly even though its per-step error is smaller.

Everything here is dependency-free (pure Python stdlib) so it runs anywhere.
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

# A "state" is (positions, velocities), each a list of [x, y, z] vectors.
Vec = List[float]
State = Tuple[List[Vec], List[Vec]]
AccelFn = Callable[[List[Vec]], List[Vec]]


def _axpy(a: float, x: List[Vec], y: List[Vec]) -> List[Vec]:
    """Return y + a*x for lists of 3-vectors (out-of-place)."""
    return [[y[i][k] + a * x[i][k] for k in range(3)] for i in range(len(y))]


def velocity_verlet(pos: List[Vec], vel: List[Vec], accel: AccelFn, dt: float) -> State:
    """Second-order symplectic integrator (a.k.a. leapfrog in kick-drift-kick form).

    This is *the* workhorse of gravitational dynamics. It is time-reversible and
    symplectic, so energy error stays bounded and oscillatory forever.
    """
    a0 = accel(pos)
    # drift half + kick + drift half, arranged as the standard KDK form:
    vel_half = _axpy(0.5 * dt, a0, vel)
    pos_new = _axpy(dt, vel_half, pos)
    a1 = accel(pos_new)
    vel_new = _axpy(0.5 * dt, a1, vel_half)
    return pos_new, vel_new


# Forest & Ruth (1990) fourth-order symplectic coefficients.
_FR_CBRT2 = 2.0 ** (1.0 / 3.0)
_FR_W1 = 1.0 / (2.0 - _FR_CBRT2)
_FR_W0 = -_FR_CBRT2 * _FR_W1
_FR_C = [0.5 * _FR_W1, 0.5 * (_FR_W0 + _FR_W1), 0.5 * (_FR_W0 + _FR_W1), 0.5 * _FR_W1]
_FR_D = [_FR_W1, _FR_W0, _FR_W1]


def forest_ruth(pos: List[Vec], vel: List[Vec], accel: AccelFn, dt: float) -> State:
    """Fourth-order symplectic integrator (Forest-Ruth). Bounded energy error,
    with a much smaller amplitude than velocity-Verlet at the same step size."""
    for i in range(3):
        pos = _axpy(_FR_C[i] * dt, vel, pos)
        a = accel(pos)
        vel = _axpy(_FR_D[i] * dt, a, vel)
    pos = _axpy(_FR_C[3] * dt, vel, pos)
    return pos, vel


def rk4(pos: List[Vec], vel: List[Vec], accel: AccelFn, dt: float) -> State:
    """Classic fourth-order Runge-Kutta. High local accuracy but NOT symplectic,
    so total energy drifts secularly on long integrations — included as the foil."""
    def deriv(p: List[Vec], v: List[Vec]) -> State:
        return v, accel(p)

    k1p, k1v = deriv(pos, vel)
    k2p, k2v = deriv(_axpy(0.5 * dt, k1p, pos), _axpy(0.5 * dt, k1v, vel))
    k3p, k3v = deriv(_axpy(0.5 * dt, k2p, pos), _axpy(0.5 * dt, k2v, vel))
    k4p, k4v = deriv(_axpy(dt, k3p, pos), _axpy(dt, k3v, vel))

    pos_new = [
        [pos[i][k] + dt / 6.0 * (k1p[i][k] + 2 * k2p[i][k] + 2 * k3p[i][k] + k4p[i][k])
         for k in range(3)]
        for i in range(len(pos))
    ]
    vel_new = [
        [vel[i][k] + dt / 6.0 * (k1v[i][k] + 2 * k2v[i][k] + 2 * k3v[i][k] + k4v[i][k])
         for k in range(3)]
        for i in range(len(vel))
    ]
    return pos_new, vel_new


INTEGRATORS = {
    "verlet": velocity_verlet,
    "forest_ruth": forest_ruth,
    "rk4": rk4,
}
