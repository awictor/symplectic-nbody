"""Relativistic beaming: why one jet is bright and its twin invisible.

Radiation from a source moving near the speed of light is not emitted the same in all
directions. Two relativistic effects conspire: aberration sweeps the emission forward
into a narrow cone of half-angle ~1/gamma (the "headlight effect"), and the Doppler
shift boosts both the photon energy and the arrival rate. Together they concentrate the
observed flux by powers of the Doppler factor

    D = 1 / (gamma (1 - beta cos theta)),

with beta = v/c, gamma the Lorentz factor, and theta the angle between the motion and
the line of sight. The observed flux of a moving blob scales as

    S_obs = D^(3 + alpha) S_emit    (alpha the spectral index, ~ D^3-D^4),

so an approaching jet is hugely brightened and a receding one dimmed by the same
powers. That asymmetry is why active-galaxy jets like M87's look one-sided: the
counter-jet is really there, just beamed away from us. The jet-to-counterjet flux ratio

    R = ((1 + beta cos theta)/(1 - beta cos theta))^(3 + alpha)

measures the speed directly, and superluminal apparent motion is the same geometry seen
in the plane of the sky.

This module gives the Lorentz factor, the Doppler factor, the beaming flux boost, the
beaming cone half-angle, and the jet/counter-jet ratio, and reproduces the strong
one-sidedness of a relativistic jet. Dimensionless / SI. Pure stdlib; the
special-relativity companion to the relativity and synchrotron modules.
"""

from __future__ import annotations

import math


def lorentz_factor(beta: float) -> float:
    """Lorentz factor gamma = 1 / sqrt(1 - beta^2)."""
    return 1.0 / math.sqrt(1.0 - beta * beta)


def doppler_factor(beta: float, theta_rad: float) -> float:
    """Relativistic Doppler factor D = 1 / (gamma (1 - beta cos theta)). D>1 for a
    source approaching within the beaming cone, D<1 for a receding one."""
    gamma = lorentz_factor(beta)
    return 1.0 / (gamma * (1.0 - beta * math.cos(theta_rad)))


def flux_boost(beta: float, theta_rad: float, alpha: float = 0.7,
               continuous: bool = False) -> float:
    """Observed/emitted flux ratio from beaming: D^(3+alpha) for a discrete blob, or
    D^(2+alpha) for a continuous (steady) jet. alpha is the spectral index."""
    D = doppler_factor(beta, theta_rad)
    exponent = (2.0 + alpha) if continuous else (3.0 + alpha)
    return D ** exponent


def beaming_cone_halfangle(beta: float) -> float:
    """Half-angle (radians) of the forward beaming cone, ~1/gamma: the emission is
    swept into this cone by relativistic aberration."""
    return 1.0 / lorentz_factor(beta)


def jet_counterjet_ratio(beta: float, theta_rad: float, alpha: float = 0.7,
                         continuous: bool = False) -> float:
    """Flux ratio of the approaching jet to the receding counter-jet:
    ((1 + beta cos theta)/(1 - beta cos theta))^(p), p = 3+alpha (blob) or 2+alpha."""
    p = (2.0 + alpha) if continuous else (3.0 + alpha)
    mu = beta * math.cos(theta_rad)
    return ((1.0 + mu) / (1.0 - mu)) ** p


def apparent_transverse_speed(beta: float, theta_rad: float) -> float:
    """Apparent transverse speed in units of c: beta sin theta / (1 - beta cos theta).
    Exceeds 1 (superluminal) for fast, small-angle jets -- an illusion of light travel
    time, not a real faster-than-light motion."""
    return beta * math.sin(theta_rad) / (1.0 - beta * math.cos(theta_rad))
