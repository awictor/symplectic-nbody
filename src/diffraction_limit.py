"""The diffraction limit: why every lens and telescope has a resolution floor.

No optical instrument can focus light to a point. A wave passing through an aperture of
diameter D spreads by diffraction into an Airy pattern whose central disk has an angular
radius (to the first dark ring)

    theta = 1.22 lambda / D,

the Rayleigh criterion. Two point sources closer than this on the sky (or under a microscope)
blur into one -- so the aperture sets the finest detail an instrument can resolve, no matter
how good the glass. It is why telescopes are built ever larger (a bigger D means a smaller
theta), why radio dishes are enormous (long lambda needs huge D), and why a human pupil
resolves about an arcminute.

The linear resolution at a distance L is L theta, and for a microscope the smallest resolvable
feature is set by Abbe's diffraction limit d = lambda / (2 NA), where NA is the numerical
aperture -- which is why electron microscopes, using picometre de Broglie wavelengths, see
atoms while light microscopes stop at ~200 nm.

A diffraction grating turns the same physics into a spectrometer: N lines of spacing g
disperse wavelengths by d(sin theta) = m lambda / g and resolve them with a resolving power
R = lambda / dlambda = m N, so more lines and higher orders separate finer spectral detail.

This module gives the Rayleigh angular resolution, the linear resolution at a distance, the
Abbe microscope limit, the grating diffraction angle and resolving power, and the aperture
needed for a target resolution, and reproduces Hubble's ~0.05-arcsec resolution and the
~200 nm light-microscope floor. SI units, angles in radians unless _deg. Pure stdlib; the
wave-optics companion to the Bragg and de Broglie notes.
"""

from __future__ import annotations

import math


def rayleigh_angle(wavelength: float, aperture: float) -> float:
    """Rayleigh angular resolution theta = 1.22 lambda / D (rad): the smallest angle between
    two point sources an aperture of diameter D can separate."""
    return 1.22 * wavelength / aperture


def rayleigh_angle_arcsec(wavelength: float, aperture: float) -> float:
    """Rayleigh angular resolution in arcseconds."""
    return math.degrees(rayleigh_angle(wavelength, aperture)) * 3600.0


def linear_resolution(wavelength: float, aperture: float, distance: float) -> float:
    """Smallest separation (m) resolvable at a distance L: L * 1.22 lambda / D."""
    return distance * rayleigh_angle(wavelength, aperture)


def abbe_limit(wavelength: float, numerical_aperture: float) -> float:
    """Abbe microscope diffraction limit d = lambda / (2 NA) (m): the finest feature a lens of
    numerical aperture NA can resolve. ~200 nm for visible light, NA ~ 1.4."""
    return wavelength / (2.0 * numerical_aperture)


def aperture_for_resolution(wavelength: float, theta: float) -> float:
    """Aperture diameter D = 1.22 lambda / theta (m) needed to reach an angular resolution
    theta. Inverts rayleigh_angle."""
    return 1.22 * wavelength / theta


def grating_angle(wavelength: float, line_spacing: float, order: int = 1) -> float:
    """Diffraction angle theta for a grating of line spacing g: sin(theta) = m lambda / g,
    so theta = arcsin(m lambda / g). Raises if m lambda > g (order does not exist)."""
    s = order * wavelength / line_spacing
    if s > 1.0:
        raise ValueError("grating order does not exist: m*lambda exceeds line spacing")
    return math.asin(s)


def grating_resolving_power(order: int, n_lines: int) -> float:
    """Grating resolving power R = lambda / dlambda = m N: the product of the order and the
    number of illuminated lines. Higher order or more lines separate finer wavelengths."""
    return float(order * n_lines)
