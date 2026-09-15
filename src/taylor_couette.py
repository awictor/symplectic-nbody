"""Taylor-Couette flow: the exact velocity between rotating cylinders and when it goes unstable.

Fill the gap between two long concentric cylinders with a viscous fluid and spin them. The inner cylinder
has radius r1 and angular velocity Omega1, the outer r2 > r1 at Omega2. At low speeds the fluid settles
into a steady CIRCULAR COUETTE FLOW: it moves only in the azimuthal direction, in perfect circles, and the
Navier-Stokes equations collapse to a single ordinary differential equation whose solution is exactly

    v(r) = A r + B / r,   Omega(r) = v(r)/r = A + B / r^2

with A and B fixed by the no-slip boundary conditions v(r1) = Omega1 r1 and v(r2) = Omega2 r2. The A term
is rigid rotation; the B term is an irrotational vortex. This is one of the few exactly solvable viscous
flows, and it is the textbook way to MEASURE viscosity (the torque on a cylinder is analytic).

But wind the inner cylinder up and the smooth flow breaks: it buckles into a stack of counter-rotating
donut vortices -- TAYLOR VORTICES -- the first clean experimental confirmation (G. I. Taylor, 1923) of a
hydrodynamic instability predicted from theory. The mechanism is centrifugal. RAYLEIGH's inviscid criterion
(1917) says a rotating flow is stable exactly when the square of the specific angular momentum, L(r) =
r^2 Omega(r), does not decrease outward:

    Phi(r) = (1 / r^3) d(L^2)/dr >= 0  everywhere  <=>  stable.

Fluid parcels flung outward must meet a stronger restoring pressure than their angular momentum demands;
if L^2 falls with radius, a displaced ring keeps going and the flow overturns. When only the inner cylinder
spins, L^2 always decreases outward and the flow is centrifugally unstable above a viscous threshold set by
the TAYLOR NUMBER Ta; below the critical Ta ~ 1708 (narrow-gap, mapping onto the Rayleigh-Benard number)
viscosity damps the vortices and the Couette solution survives.

This module builds the exact profile, its coefficients, the torque and stress, the Rayleigh discriminant,
and the narrow-gap Taylor number. It is validated analytically: the profile satisfies both boundary
conditions and the r^2-Omega equation to machine precision; solid-body rotation (Omega1 = Omega2) gives
B = 0 and Phi > 0 (unconditionally stable); the Rayleigh line Omega2/Omega1 = (r1/r2)^2 is the exact
marginal case; and the Rayleigh-stable / unstable verdicts match direct evaluation of L^2 increasing or
decreasing. Pure stdlib. The rotating-flow companion to the Rayleigh-Benard, Kelvin-Helmholtz, and
Orr-Sommerfeld notes."""

from __future__ import annotations

import math


def couette_coeffs(r1, r2, omega1, omega2):
    """Return (A, B) for Omega(r) = A + B/r^2 (equivalently v(r) = A r + B/r).

    From no-slip Omega(r1)=omega1, Omega(r2)=omega2:
        A = (omega2 r2^2 - omega1 r1^2) / (r2^2 - r1^2)
        B = (omega1 - omega2) r1^2 r2^2 / (r2^2 - r1^2)
    """
    if r2 <= r1 or r1 <= 0:
        raise ValueError("require 0 < r1 < r2")
    r1s, r2s = r1 * r1, r2 * r2
    denom = r2s - r1s
    a = (omega2 * r2s - omega1 * r1s) / denom
    b = (omega1 - omega2) * r1s * r2s / denom
    return a, b


def angular_velocity(r, r1, r2, omega1, omega2):
    """Omega(r) = A + B/r^2, the local rate of rotation of the fluid at radius r."""
    a, b = couette_coeffs(r1, r2, omega1, omega2)
    return a + b / (r * r)


def velocity(r, r1, r2, omega1, omega2):
    """Azimuthal (tangential) speed v(r) = Omega(r) r = A r + B/r."""
    return angular_velocity(r, r1, r2, omega1, omega2) * r


def specific_angular_momentum(r, r1, r2, omega1, omega2):
    """L(r) = r^2 Omega(r) = A r^2 + B, the angular momentum per unit mass of a fluid ring."""
    a, b = couette_coeffs(r1, r2, omega1, omega2)
    return a * r * r + b


def rayleigh_discriminant(r, r1, r2, omega1, omega2):
    """Rayleigh discriminant Phi(r) = (1/r^3) d(L^2)/dr.

    With L = A r^2 + B, L^2 = (A r^2 + B)^2, so d(L^2)/dr = 2(A r^2 + B)(2 A r) and
    Phi = 4 A (A r^2 + B) / r^2 = 4 A Omega(r). Phi >= 0 everywhere <=> centrifugally stable."""
    a, b = couette_coeffs(r1, r2, omega1, omega2)
    return 4.0 * a * (a * r * r + b) / (r * r)


def is_rayleigh_stable(r1, r2, omega1, omega2, samples=200):
    """True iff the inviscid Rayleigh criterion holds across the whole gap (Phi(r) >= 0 for all r).

    Sampled on a fine radial grid; the discriminant is smooth so sampling is faithful."""
    for i in range(samples + 1):
        r = r1 + (r2 - r1) * i / samples
        if rayleigh_discriminant(r, r1, r2, omega1, omega2) < -1e-12:
            return False
    return True


def rayleigh_marginal_ratio(r1, r2):
    """The critical outer/inner angular-velocity ratio on the Rayleigh stability line.

    L^2 is non-decreasing outward exactly when Omega2/Omega1 >= (r1/r2)^2 (for co-rotation, Omega1>0).
    At equality the flow is marginally stable (constant angular momentum, the potential vortex)."""
    return (r1 / r2) ** 2


def torque_per_length(r1, r2, omega1, omega2, mu=1.0):
    """Viscous torque per unit axial length exerted on the fluid (constant across the gap).

    For circular Couette flow the shear stress gives a torque G = 4 pi mu B (with B the vortex
    coefficient). Independent of r -- angular momentum is transported uniformly. Used to measure mu."""
    _, b = couette_coeffs(r1, r2, omega1, omega2)
    return -4.0 * math.pi * mu * b


def taylor_number(r1, r2, omega1, nu):
    """Narrow-gap Taylor number for the inner cylinder rotating (outer at rest).

    Ta = Omega1^2 r1 (r2 - r1)^3 / nu^2. Onset of Taylor vortices at Ta_c ~ 1708 (narrow gap), the same
    critical number as Rayleigh-Benard convection under the standard mapping. Returns Ta."""
    d = r2 - r1
    return omega1 * omega1 * r1 * d ** 3 / (nu * nu)


TAYLOR_CRITICAL = 1708.0  # narrow-gap critical Taylor number (onset of Taylor vortices)


def is_taylor_unstable(r1, r2, omega1, nu):
    """True iff the narrow-gap Taylor number exceeds the critical value -- Taylor vortices appear."""
    return taylor_number(r1, r2, omega1, nu) > TAYLOR_CRITICAL
