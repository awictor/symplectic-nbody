"""Chandrasekhar dynamical friction: gravitational drag through a star field.

A massive body moving through a sea of lighter stars gravitationally focuses
them into a wake behind it. The wake's pull decelerates the body -- dynamical
friction. Chandrasekhar (1943) gave the drag:

    dv/dt = -4 pi G^2 M rho ln(Lambda) / v^2 * [erf(X) - 2X/sqrt(pi) e^{-X^2}] * v_hat,

where rho is the background density, ln(Lambda) the Coulomb logarithm, and
X = v / (sqrt(2) sigma) with sigma the stars' velocity dispersion. The force
grows with M^2 and falls as 1/v^2, so heavy, slow objects sink fastest.

Consequences this module reproduces:
  * a satellite or globular cluster orbiting a galaxy spirals inward and merges,
    on a time that scales as 1/M (heavier sinks faster);
  * massive black holes sink to a galaxy's centre; globular clusters are eroded.

Isothermal-halo units are generic (set G); the sinking-time formula uses the
standard singular-isothermal-sphere result. Pure stdlib.
"""

from __future__ import annotations

import math

G = 6.67430e-11


def _erf(x: float) -> float:
    """Error function (Abramowitz & Stegun 7.1.26), max error ~1.5e-7."""
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * math.exp(-x * x)
    return sign * y


def friction_acceleration(M: float, v: float, rho: float, sigma: float,
                          ln_lambda: float = 5.0) -> float:
    """Magnitude of the Chandrasekhar deceleration (opposing the motion) for a
    body of mass M moving at speed v through density rho with dispersion sigma."""
    X = v / (math.sqrt(2.0) * sigma)
    maxwell = _erf(X) - 2.0 * X / math.sqrt(math.pi) * math.exp(-X * X)
    return 4.0 * math.pi * G * G * M * rho * ln_lambda / (v * v) * maxwell


def sinking_time(M: float, r0: float, v_circ: float, ln_lambda: float = 5.0) -> float:
    """Time for a body of mass M on a circular orbit of radius r0 in a singular
    isothermal halo (circular speed v_circ) to spiral to the centre:

        t = 1.17 r0^2 v_circ / (G M ln Lambda)   (Binney & Tremaine).

    Scales as 1/M: heavier objects sink faster."""
    return 1.17 * r0 * r0 * v_circ / (G * M * ln_lambda)
