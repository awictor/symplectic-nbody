"""Gravitational lensing by a point mass: bending light with gravity.

A mass deflects a passing light ray by the general-relativistic angle

    alpha = 4 G M / (c^2 b)          (b = impact parameter)

-- exactly twice the Newtonian value, the prediction Eddington confirmed at the
1919 eclipse. For a source, lens, and observer nearly in line this produces
multiple images, an Einstein ring, and magnification -- the basis of
strong lensing and microlensing planet searches.

For a point-mass lens the geometry reduces to the lens equation

    beta = theta - theta_E^2 / theta

where beta is the true source angle, theta the image angle, and theta_E the
Einstein radius

    theta_E = sqrt( 4 G M / c^2 * D_LS / (D_L D_S) ).

This module solves the lens equation for the two images, their magnifications,
and reproduces the bare deflection law. Angles are in radians. Pure stdlib.
"""

from __future__ import annotations

import math
from typing import List, Tuple

G = 6.67430e-11          # SI
C = 2.99792458e8         # m/s


def deflection_angle(M: float, b: float) -> float:
    """GR light-bending angle alpha = 4 G M / (c^2 b) for impact parameter b."""
    return 4.0 * G * M / (C * C * b)


def einstein_radius(M: float, D_L: float, D_S: float, D_LS: float) -> float:
    """Einstein angular radius (radians) for lens mass M and angular-diameter
    distances observer-lens D_L, observer-source D_S, lens-source D_LS."""
    return math.sqrt(4.0 * G * M / (C * C) * D_LS / (D_L * D_S))


def image_positions(beta: float, theta_E: float) -> Tuple[float, float]:
    """Solve the point-mass lens equation beta = theta - theta_E^2/theta for the
    two image angles. Returns (theta_plus, theta_minus); theta_minus is negative
    (image on the opposite side). Valid for beta >= 0."""
    disc = math.sqrt(beta * beta + 4.0 * theta_E * theta_E)
    theta_plus = 0.5 * (beta + disc)
    theta_minus = 0.5 * (beta - disc)
    return theta_plus, theta_minus


def magnification(theta: float, theta_E: float) -> float:
    """Signed magnification of an image at angle theta:
    mu = 1 / (1 - (theta_E/theta)^4). Negative => parity-flipped image."""
    u = (theta_E / theta) ** 4
    return 1.0 / (1.0 - u)


def total_magnification(beta: float, theta_E: float) -> float:
    """Total (unsigned) magnification of a point source -- the microlensing
    light-curve amplitude. For u = beta/theta_E:
        A = (u^2 + 2) / (u sqrt(u^2 + 4))."""
    u = beta / theta_E
    if u == 0.0:
        return float("inf")
    return (u * u + 2.0) / (u * math.sqrt(u * u + 4.0))


def image_separation(beta: float, theta_E: float) -> float:
    """Angular separation of the two images."""
    tp, tm = image_positions(beta, theta_E)
    return tp - tm  # tm is negative, so this is |tp| + |tm|
