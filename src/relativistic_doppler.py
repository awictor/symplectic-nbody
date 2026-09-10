"""The relativistic Doppler effect: colour shifts of fast-moving light sources.

A moving light source shifts in frequency by two combined effects: the classical
Doppler shift from the changing path length, and relativistic time dilation of the
source's clock. For motion directly toward or away at speed beta = v/c the observed
frequency is

    f_obs / f_src = sqrt((1 - beta)/(1 + beta))   (receding, redshift),
                  = sqrt((1 + beta)/(1 - beta))   (approaching, blueshift),

and the wavelength ratio is the reciprocal. The redshift z = lambda_obs/lambda_src - 1
is the quantity astronomers report.

The purely relativistic surprise is the TRANSVERSE Doppler shift: even when the source
moves exactly perpendicular to the line of sight (no classical shift at all), its
clock still runs slow, so the light is redshifted by a factor 1/gamma. Ives and Stilwell
measured exactly this in 1938, confirming time dilation directly.

For general angle theta (between the source velocity and the line of sight, in the
observer frame) the shift is f_obs/f_src = 1/(gamma(1 - beta cos theta)) -- the same
Doppler factor that beams jets. This module gives the longitudinal and transverse
shifts, the redshift z, the velocity implied by a measured z, and the general-angle
factor, and reproduces the receding/approaching shifts and the transverse time-dilation
redshift. Dimensionless. Pure stdlib; the special-relativistic companion to the beaming
and cosmology modules.
"""

from __future__ import annotations

import math


def lorentz_factor(beta: float) -> float:
    """Lorentz factor gamma = 1/sqrt(1 - beta^2)."""
    return 1.0 / math.sqrt(1.0 - beta * beta)


def frequency_ratio_radial(beta: float, approaching: bool = False) -> float:
    """Observed/source frequency for radial motion:
    receding sqrt((1-beta)/(1+beta)) < 1; approaching sqrt((1+beta)/(1-beta)) > 1."""
    if approaching:
        return math.sqrt((1.0 + beta) / (1.0 - beta))
    return math.sqrt((1.0 - beta) / (1.0 + beta))


def wavelength_ratio_radial(beta: float, approaching: bool = False) -> float:
    """Observed/source wavelength for radial motion (reciprocal of the frequency
    ratio)."""
    return 1.0 / frequency_ratio_radial(beta, approaching)


def redshift_radial(beta: float, approaching: bool = False) -> float:
    """Redshift z = lambda_obs/lambda_src - 1 for radial motion. Positive (redshift)
    receding, negative (blueshift) approaching."""
    return wavelength_ratio_radial(beta, approaching) - 1.0


def transverse_frequency_ratio(beta: float) -> float:
    """Transverse Doppler frequency ratio f_obs/f_src = 1/gamma < 1 -- a pure
    time-dilation redshift with no classical component (source moving perpendicular)."""
    return 1.0 / lorentz_factor(beta)


def transverse_redshift(beta: float) -> float:
    """Transverse redshift z = gamma - 1 (the wavelength is stretched by gamma)."""
    return lorentz_factor(beta) - 1.0


def frequency_ratio_angle(beta: float, theta_rad: float) -> float:
    """General relativistic Doppler ratio f_obs/f_src = 1/(gamma(1 - beta cos theta)),
    with theta the angle between the source's velocity and the line of sight in the
    observer frame. theta=0 approaching, pi receding, pi/2 transverse."""
    return 1.0 / (lorentz_factor(beta) * (1.0 - beta * math.cos(theta_rad)))


def beta_from_redshift(z: float) -> float:
    """Recover the radial recession speed beta from a measured (special-relativistic)
    redshift z: beta = ((1+z)^2 - 1)/((1+z)^2 + 1)."""
    r = (1.0 + z) ** 2
    return (r - 1.0) / (r + 1.0)
