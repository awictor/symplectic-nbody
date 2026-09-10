"""Thin-film interference: the colours of soap bubbles and oil slicks.

When light hits a thin transparent film, part reflects off the top surface and part off the
bottom. The two reflected waves travel different distances and recombine; whether they add or
cancel depends on the film thickness, so a film only nanometres thick paints itself in colour.
The extra optical path for near-normal incidence is 2 n t (down and back through a film of
index n and thickness t), but there is a subtlety: a reflection off a denser medium flips the
wave by half a wavelength. For a film with air on both sides (a soap bubble, index n over
air), only the top reflection flips, so the interference condition is

    constructive:  2 n t = (m - 1/2) lambda
    destructive:   2 n t = m lambda,

and the reverse when the film sits on a denser substrate (like oil on water or a coating on
glass). This half-wave shift is why a very thin soap film looks black just before it bursts:
2 n t -> 0 gives destructive interference at every wavelength.

The same physics is engineered into anti-reflection coatings: a quarter-wave layer
(t = lambda / (4 n)) of the right index makes the two reflections cancel, killing glare on
lenses and boosting solar-cell absorption. Newton's rings -- the concentric fringes in the
air gap between a lens and a flat -- are thin-film interference with a thickness that grows
with radius.

This module gives the constructive/destructive wavelengths for a film, the quarter-wave
anti-reflection thickness and its ideal index, the film thickness for a target reflected
colour, and Newton's-ring radii, and reproduces the soap-bubble colours and the MgF2 lens
coating. SI units. Pure stdlib; the wave-optics companion to the Snell and Bragg notes.
"""

from __future__ import annotations

import math


def optical_path(n_film: float, thickness: float) -> float:
    """Extra optical path 2 n t (m) for light making one round trip through the film at
    near-normal incidence."""
    return 2.0 * n_film * thickness


def constructive_wavelength(n_film: float, thickness: float, order: int = 1,
                            half_wave_shift: bool = True) -> float:
    """Wavelength (m) that reflects constructively (bright) off a film. With a single
    half-wave shift (film over air, e.g. soap bubble): 2 n t = (m - 1/2) lambda. Order m>=1."""
    if half_wave_shift:
        return optical_path(n_film, thickness) / (order - 0.5)
    return optical_path(n_film, thickness) / order


def destructive_wavelength(n_film: float, thickness: float, order: int = 1,
                           half_wave_shift: bool = True) -> float:
    """Wavelength (m) that reflects destructively (dark) off a film. With a single half-wave
    shift: 2 n t = m lambda. Order m>=1."""
    if half_wave_shift:
        return optical_path(n_film, thickness) / order
    return optical_path(n_film, thickness) / (order - 0.5)


def antireflection_thickness(wavelength: float, n_coating: float) -> float:
    """Quarter-wave anti-reflection coating thickness t = lambda / (4 n) (m): makes the top
    and bottom reflections cancel at that wavelength."""
    return wavelength / (4.0 * n_coating)


def ideal_ar_index(n_substrate: float, n_outside: float = 1.0) -> float:
    """Ideal anti-reflection coating index n = sqrt(n_outside n_substrate), which makes the
    two reflection amplitudes equal so they cancel completely. ~1.23 for glass in air (MgF2
    at 1.38 is the practical choice)."""
    return math.sqrt(n_outside * n_substrate)


def newton_ring_radius(order: int, wavelength: float, lens_radius: float,
                       bright: bool = False) -> float:
    """Radius (m) of the m-th Newton's ring in the air gap under a lens of curvature radius R.
    Dark rings: r = sqrt(m lambda R); bright rings: r = sqrt((m - 1/2) lambda R)."""
    m = order - 0.5 if bright else order
    return math.sqrt(m * wavelength * lens_radius)
