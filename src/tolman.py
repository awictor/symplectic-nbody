"""Tolman surface-brightness dimming: the sharpest test that the universe expands.

An object's surface brightness -- flux per unit solid angle -- is distance-independent
in a static, Euclidean universe: a galaxy twice as far has 1/4 the flux but also 1/4
the angular area, so its brightness per square arcsecond is unchanged. That is why the
Moon looks equally bright whether high or low in the sky.

In an EXPANDING universe this fails, and fails hard. Four factors of (1+z) pile up:

  * photon energy is redshifted           -> (1+z)^-1
  * photon arrival rate is time-dilated    -> (1+z)^-1
  * the luminosity distance stretches by (1+z) relative to the angular-diameter
    distance, and surface brightness goes as (D_A / D_L)^2 = (1+z)^-4 ... combining
    all effects,

    SB_observed / SB_emitted = (1 + z)^(-4).

So a galaxy at z = 1 is dimmed per unit area by a factor of 16; at z = 3, by 256. This
Tolman test cleanly distinguishes an expanding universe (dimming ~ (1+z)^-4) from a
static "tired-light" universe (which predicts only (1+z)^-1), and observations confirm
the expanding-universe exponent -- one of the most direct pieces of evidence that the
redshift is genuine expansion, not photons losing energy en route.

This module gives the dimming factor, its expression in magnitudes, the tired-light
comparison, and the exponent that observations must match, and reproduces the 16x-at-
z=1 dimming. Dimensionless. Pure stdlib; the cosmological-test companion to the
Friedmann and distances modules.
"""

from __future__ import annotations

import math


def dimming_factor(z: float, exponent: float = 4.0) -> float:
    """Surface-brightness dimming ratio SB_obs / SB_emit = (1+z)^(-exponent).
    exponent=4 for an expanding (FLRW) universe."""
    return (1.0 + z) ** (-exponent)


def dimming_magnitudes(z: float, exponent: float = 4.0) -> float:
    """Surface-brightness dimming expressed in magnitudes per unit area:
    dmu = 2.5 exponent log10(1+z) = -2.5 log10(dimming factor)."""
    return -2.5 * math.log10(dimming_factor(z, exponent))


def tired_light_factor(z: float) -> float:
    """Surface-brightness ratio in a static 'tired-light' universe: only the single
    energy-loss factor (1+z)^-1, with no time dilation or geometric stretching."""
    return dimming_factor(z, exponent=1.0)


def expanding_vs_tired(z: float) -> float:
    """Ratio of expanding-universe dimming to tired-light dimming at redshift z:
    (1+z)^-4 / (1+z)^-1 = (1+z)^-3. How much fainter the expanding prediction is."""
    return dimming_factor(z, 4.0) / tired_light_factor(z)


def exponent_from_observation(z: float, observed_ratio: float) -> float:
    """Recover the dimming exponent implied by an observed SB ratio at redshift z:
    n = -log(observed_ratio) / log(1+z). Should come out ~4 for expansion."""
    return -math.log(observed_ratio) / math.log(1.0 + z)
