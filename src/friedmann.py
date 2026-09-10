"""Friedmann cosmology: the expansion history of the universe.

The scale factor a(t) (with a=1 today) evolves under the Friedmann equation

    (a_dot / a)^2 = H0^2 [ Omega_r/a^4 + Omega_m/a^3 + Omega_k/a^2 + Omega_L ]

where the density parameters (radiation, matter, curvature, dark energy) sum to 1
for consistency: Omega_r + Omega_m + Omega_k + Omega_L = 1. Each component dilutes
differently as the universe expands -- radiation as a^-4, matter as a^-3, dark
energy not at all -- so the universe passes through radiation-, matter-, and
dark-energy-dominated eras with distinct expansion laws:

    radiation era : a(t) ~ t^{1/2}
    matter era    : a(t) ~ t^{2/3}
    Lambda era    : a(t) ~ exp(H t)   (accelerating)

This module integrates a(t) forward and backward from today, reproduces those
scalings, and computes the age of the universe as a look-back integral. Units:
time in 1/H0 (so the answer is dimensionless "Hubble times"). Pure stdlib.
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple


class Cosmology:
    def __init__(self, Omega_r: float = 0.0, Omega_m: float = 0.3,
                 Omega_L: float = 0.7, H0: float = 1.0):
        self.Omega_r = Omega_r
        self.Omega_m = Omega_m
        self.Omega_L = Omega_L
        # curvature closes the sum to 1
        self.Omega_k = 1.0 - Omega_r - Omega_m - Omega_L
        self.H0 = H0

    def E(self, a: float) -> float:
        """Dimensionless Hubble rate E(a) = H(a)/H0 = a_dot/(a H0) ... actually
        H(a)/H0; a_dot = a H0 E(a)."""
        return math.sqrt(self.Omega_r / a ** 4 + self.Omega_m / a ** 3
                         + self.Omega_k / a ** 2 + self.Omega_L)

    def a_dot(self, a: float) -> float:
        """da/dt = a H0 E(a)."""
        return a * self.H0 * self.E(a)

    def integrate_forward(self, a0: float = 1e-3, t_max: float = 3.0,
                          dt: float = 1e-4) -> Tuple[List[float], List[float]]:
        """Integrate a(t) forward from a small a0. Returns (times, a) with t=0 at
        a=a0. Uses RK4 on da/dt = a H0 E(a)."""
        ts, as_ = [0.0], [a0]
        a, t = a0, 0.0
        n = int(t_max / dt)
        for _ in range(n):
            k1 = self.a_dot(a)
            k2 = self.a_dot(a + 0.5 * dt * k1)
            k3 = self.a_dot(a + 0.5 * dt * k2)
            k4 = self.a_dot(a + dt * k3)
            a += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
            ts.append(t); as_.append(a)
            if a > 50.0:
                break
        return ts, as_

    def age(self, a_from: float = 1e-6, a_to: float = 1.0, n: int = 200000) -> float:
        """Age of the universe (time since a_from) as the integral
        t = integral_{a_from}^{a_to} da / (a H0 E(a)), by the trapezoid rule."""
        loga0, loga1 = math.log(a_from), math.log(a_to)
        total = 0.0
        prev_a = a_from
        prev_int = 1.0 / (self.a_dot(prev_a))
        for i in range(1, n + 1):
            a = math.exp(loga0 + (loga1 - loga0) * i / n)
            cur_int = 1.0 / self.a_dot(a)
            # integrate in a: dt = da/a_dot; use trapezoid in a
            total += 0.5 * (prev_int + cur_int) * (a - prev_a)
            prev_a, prev_int = a, cur_int
        return total


def local_slope(ts: List[float], as_: List[float], i: int, half: int = 500) -> float:
    """Local log-log slope d ln a / d ln t near sample i (the expansion exponent
    a ~ t^n)."""
    lo = max(1, i - half)
    hi = min(len(ts) - 1, i + half)
    lt = [math.log(ts[j]) for j in range(lo, hi) if ts[j] > 0]
    la = [math.log(as_[j]) for j in range(lo, hi) if ts[j] > 0]
    m = len(lt)
    mt, ma = sum(lt) / m, sum(la) / m
    num = sum((lt[k] - mt) * (la[k] - ma) for k in range(m))
    den = sum((lt[k] - mt) ** 2 for k in range(m))
    return num / den
