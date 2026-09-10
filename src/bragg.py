"""Bragg diffraction: reading a crystal with waves.

Shine X-rays (or neutrons, or electrons) on a crystal and most pass through, but at special
angles the waves scattered from successive planes of atoms add up in phase and flash out a
bright reflected beam. The condition, found by the Braggs in 1913, is that the extra path
between planes spaced d apart be a whole number of wavelengths:

    n lambda = 2 d sin(theta),

where theta is the glancing angle (measured from the plane, not the normal) and n the
diffraction order. That single equation is the foundation of X-ray crystallography -- turning
a pattern of diffraction spots into the atomic structure of everything from table salt to
DNA to proteins. It works because the wavelength must be comparable to the ~0.1-0.5 nm
spacing of atoms, which is exactly the X-ray range.

For a cubic crystal of lattice constant a the planes labelled by Miller indices (h,k,l) are
spaced d = a / sqrt(h^2 + k^2 + l^2), so each set of planes gives its own family of Bragg
angles. Bragg's law also caps what you can see: since sin(theta) <= 1, no reflection exists
unless lambda <= 2d, so a wavelength longer than twice the spacing diffracts from nothing.

This module gives the Bragg angle for a given order and spacing, the plane spacing (including
the cubic Miller-index formula), the wavelength or order from a measured angle, and the
maximum observable order, and reproduces the ~0.15 nm copper-K-alpha reflections from a
silicon-scale lattice. SI units, angles in radians unless _deg. Pure stdlib; the wave-optics
companion to the de Broglie and photoelectric notes.
"""

from __future__ import annotations

import math


def bragg_angle(wavelength: float, spacing: float, order: int = 1) -> float:
    """Bragg glancing angle theta (rad) from n lambda = 2 d sin(theta):
    theta = arcsin(n lambda / (2 d)). Raises if n lambda > 2 d (no such reflection)."""
    s = order * wavelength / (2.0 * spacing)
    if s > 1.0:
        raise ValueError("no Bragg reflection: n*lambda exceeds 2d")
    return math.asin(s)


def bragg_angle_deg(wavelength: float, spacing: float, order: int = 1) -> float:
    """Bragg angle in degrees."""
    return math.degrees(bragg_angle(wavelength, spacing, order))


def plane_spacing(wavelength: float, theta: float, order: int = 1) -> float:
    """Interplanar spacing d = n lambda / (2 sin theta) (m) from a measured Bragg angle."""
    return order * wavelength / (2.0 * math.sin(theta))


def cubic_spacing(a: float, h: int, k: int, l: int) -> float:
    """Spacing d = a / sqrt(h^2 + k^2 + l^2) (m) of the (h,k,l) planes in a cubic lattice of
    constant a."""
    return a / math.sqrt(h * h + k * k + l * l)


def wavelength_from_angle(spacing: float, theta: float, order: int = 1) -> float:
    """Wavelength lambda = 2 d sin(theta) / n (m) that produces a reflection at angle theta.
    Inverts bragg_angle."""
    return 2.0 * spacing * math.sin(theta) / order


def max_order(wavelength: float, spacing: float) -> int:
    """Highest diffraction order observable, floor(2 d / lambda), since sin(theta) <= 1.
    Zero if lambda > 2 d (nothing diffracts)."""
    return int(2.0 * spacing / wavelength)
