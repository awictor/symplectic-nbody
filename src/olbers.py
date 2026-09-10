"""Olbers' paradox: why the night sky is dark.

In an infinite, eternal, static universe uniformly filled with stars, every line of
sight would eventually end on a stellar surface, and the whole sky would blaze as
bright as the Sun. The night sky is dark -- so at least one of those assumptions is
wrong. That is Olbers' paradox, and its resolution is a genuine clue to cosmology.

The bookkeeping: a thin shell of stars at distance r contributes brightness
independent of r (each star dims as 1/r^2 but the shell holds ~r^2 more of them), so
summing shells to infinity diverges. The finite piece is the mean free path to a star,

    lambda = 1 / (n sigma),

with n the number density of stars and sigma their cross-sectional area. Only out to
~lambda do stars tile the sky; the total covered fraction after distance R is
1 - exp(-R/lambda). For realistic star densities lambda is ~10^23 light-years -- vastly
larger than the ~10^10-light-year distance light can have travelled since the Big Bang.

So the real resolution is the FINITE AGE of the universe (and, secondarily, the
redshifting of distant light): we simply cannot see far enough for the stars to tile
the sky. Every line of sight would eventually hit a star, but "eventually" is far
beyond the cosmic horizon. This module gives the mean free path to a star, the sky
covering fraction out to a distance, the horizon distance, and the fraction of the sky
actually covered within the observable universe, and reproduces the darkness of night.
SI-friendly astronomical units. Pure stdlib; the cosmology-paradox companion to the
Friedmann and Tolman modules.
"""

from __future__ import annotations

import math

LY = 9.4607e15                 # light-year in metres
R_SUN = 6.957e8
PC = 3.0856775814913673e16
HUBBLE_DISTANCE_LY = 1.38e10   # ~ c * age of universe, in light-years


def mean_free_path_to_star(n_per_m3: float, radius_m: float = R_SUN) -> float:
    """Mean free path (m) of a sight line before it hits a star: lambda = 1/(n sigma),
    sigma = pi radius^2. Beyond this, stars would tile the sky."""
    sigma = math.pi * radius_m ** 2
    return 1.0 / (n_per_m3 * sigma)


def covering_fraction(distance_m: float, mfp_m: float) -> float:
    """Fraction of the sky covered by stellar disks out to `distance`:
    1 - exp(-distance / mfp). ->1 (sky fully tiled, blazing) as distance >> mfp."""
    return 1.0 - math.exp(-distance_m / mfp_m)


def horizon_distance(age_years: float = 1.38e10) -> float:
    """Light-travel (horizon) distance c * t (m) we can actually see out to."""
    return age_years * LY / 1.0 * 1.0 if False else age_years * LY  # c*t in metres


def sky_fraction_within_horizon(n_per_m3: float, radius_m: float = R_SUN,
                                age_years: float = 1.38e10) -> float:
    """The fraction of the sky actually covered by stars within the observable
    universe: covering_fraction(horizon, mfp). Tiny -> the night sky is dark."""
    mfp = mean_free_path_to_star(n_per_m3, radius_m)
    return covering_fraction(horizon_distance(age_years), mfp)


def would_blaze_if_static(n_per_m3: float, radius_m: float = R_SUN) -> bool:
    """In an infinite static universe every line of sight hits a star, so the sky
    blazes: always True for any positive star density (the paradox). The finite age is
    what actually saves us -- see sky_fraction_within_horizon."""
    return n_per_m3 > 0.0
