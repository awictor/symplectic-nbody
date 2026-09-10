"""The Womersley number: why blood flow lags the heartbeat.

Steady flow in a pipe is Poiseuille's parabola, but blood is driven by a pulsating heart, and
a pulsating pressure does not produce a pulsating parabola. Whether the flow can keep up with
the oscillation is set by a dimensionless number John Womersley introduced in 1955,

    alpha = R sqrt(omega / nu),

the tube radius R times the square root of the angular frequency omega over the kinematic
viscosity nu. It compares the oscillation frequency to the rate at which viscosity can diffuse
momentum across the tube (the viscous penetration depth is delta = sqrt(nu/omega), so
alpha = R/delta).

Small alpha (< ~1): viscosity diffuses across the tube far faster than the flow oscillates,
so at every instant the profile is the quasi-steady Poiseuille parabola and the flow is in
phase with the pressure gradient -- the regime of small vessels and slow oscillations.

Large alpha (> ~10): the core fluid has too much inertia to follow the rapid forcing, so it
lags the pressure gradient by up to 90 degrees, the velocity profile flattens into a blunt
plug, and the shear is squeezed into a thin oscillating (Stokes) layer at the wall. The human
aorta runs at alpha ~ 15, which is why aortic flow is plug-like and lags the pressure pulse.

This module gives the Womersley number, the viscous penetration depth, the phase lag and
profile-flatness trends with alpha, the Poiseuille comparison, and the pulse-wave speed
(Moens-Korteweg), and reproduces the aorta's alpha ~ 15 and a capillary's alpha << 1. SI
units. Pure stdlib; the pulsatile-flow companion to the Reynolds and Poiseuille notes.
"""

from __future__ import annotations

import math


def womersley_number(radius: float, angular_frequency: float, nu: float) -> float:
    """Womersley number alpha = R sqrt(omega / nu): oscillation frequency vs viscous
    diffusion across the tube. <1 quasi-steady (Poiseuille), >10 inertia-dominated (plug)."""
    return radius * math.sqrt(angular_frequency / nu)


def womersley_from_heart_rate(radius: float, bpm: float, nu: float) -> float:
    """Womersley number from a heart rate in beats per minute: omega = 2 pi bpm / 60."""
    omega = 2.0 * math.pi * bpm / 60.0
    return womersley_number(radius, omega, nu)


def penetration_depth(angular_frequency: float, nu: float) -> float:
    """Viscous (Stokes) penetration depth delta = sqrt(nu / omega) (m): how far momentum
    diffuses from the wall in one oscillation. alpha = R / delta."""
    return math.sqrt(nu / angular_frequency)


def is_quasi_steady(alpha: float) -> bool:
    """True if the flow is quasi-steady (alpha < ~1): the instantaneous profile is the
    Poiseuille parabola, in phase with the pressure gradient."""
    return alpha < 1.0


def phase_lag(alpha: float) -> float:
    """Approximate phase lag (radians, 0..pi/2) of the flow behind the pressure gradient.
    ~0 for small alpha (quasi-steady), approaching pi/2 for large alpha (inertia-dominated).
    Uses a smooth interpolation lag = (pi/2) alpha^2/(alpha^2 + c) with c chosen so the
    knee sits near alpha ~ 3."""
    c = 9.0
    return (math.pi / 2.0) * alpha * alpha / (alpha * alpha + c)


def poiseuille_flow(radius: float, dP_dx: float, mu: float) -> float:
    """Steady Poiseuille volumetric flow Q = pi R^4 |dP/dx| / (8 mu) (m^3/s): the alpha -> 0
    limit that pulsatile flow reduces to."""
    return math.pi * radius ** 4 * abs(dP_dx) / (8.0 * mu)


def pulse_wave_speed(elastic_modulus: float, wall_thickness: float, radius: float,
                     rho: float = 1060.0) -> float:
    """Moens-Korteweg pulse-wave velocity c = sqrt(E h / (2 rho R)) (m/s): the speed the
    pressure pulse travels along an elastic vessel of wall modulus E, thickness h, radius R.
    ~5-10 m/s in the aorta, rising as arteries stiffen with age."""
    return math.sqrt(elastic_modulus * wall_thickness / (2.0 * rho * radius))
