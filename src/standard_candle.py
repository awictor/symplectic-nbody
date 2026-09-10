"""Standard candles: measuring cosmic distances from brightness.

If you know an object's true (absolute) luminosity, its observed (apparent) brightness
gives its distance -- the object is a "standard candle." The bookkeeping is done in
magnitudes, a logarithmic, backwards scale (brighter = smaller number) where five
magnitudes is exactly a factor of 100 in flux.

The absolute magnitude M is what a source would show at 10 pc; the apparent magnitude m
is what we see. Their difference is the distance modulus,

    m - M = 5 log10( d / 10 pc ) = 5 log10 d(pc) - 5,

which inverts to d = 10^((m - M + 5)/5) pc. A modulus of 0 means 10 pc; the Large
Magellanic Cloud's mu ~ 18.5 puts it at ~50 kpc; a galaxy at mu ~ 35 sits at ~100 Mpc.

The trick is knowing M. Cepheid variables provide it through the period-luminosity
relation Henrietta Leavitt discovered: their pulsation period tracks their luminosity,

    M_V ~ -2.81 log10(P/days) - 1.43,

so timing a Cepheid's brightness cycle reveals its absolute magnitude and hence its
distance. Type Ia supernovae (M ~ -19.3) extend the same logic to hundreds of Mpc and
built the case for cosmic acceleration. Chaining parallax -> Cepheids -> supernovae is
the cosmic distance ladder.

This module gives the distance modulus, the distance from a modulus, the Cepheid
period-luminosity relation, and the flux ratio for a magnitude difference, and
reproduces the LMC modulus and the 100-per-5-magnitudes flux rule. Astronomical units.
Pure stdlib; the photometric-distance companion to the parallax and distances modules.
"""

from __future__ import annotations

import math

PC = 3.0856775814913673e16


def distance_modulus(d_pc: float) -> float:
    """Distance modulus mu = m - M = 5 log10(d/10 pc) = 5 log10 d - 5, d in parsecs."""
    return 5.0 * math.log10(d_pc) - 5.0


def distance_from_modulus(mu: float) -> float:
    """Distance in parsecs from the distance modulus: d = 10^((mu + 5)/5)."""
    return 10.0 ** ((mu + 5.0) / 5.0)


def apparent_magnitude(M_abs: float, d_pc: float) -> float:
    """Apparent magnitude m = M + distance modulus."""
    return M_abs + distance_modulus(d_pc)


def absolute_magnitude(m_app: float, d_pc: float) -> float:
    """Absolute magnitude M = m - distance modulus (what the source shows at 10 pc)."""
    return m_app - distance_modulus(d_pc)


def flux_ratio(delta_mag: float) -> float:
    """Flux ratio for a magnitude difference: F1/F2 = 10^(-0.4 (m1 - m2)).
    Five magnitudes = a factor of 100."""
    return 10.0 ** (-0.4 * delta_mag)


def cepheid_absolute_magnitude(period_days: float, slope: float = -2.81,
                               zero_point: float = -1.43) -> float:
    """Cepheid period-luminosity relation (V band): M_V = slope log10(P) + zero_point.
    Longer-period Cepheids are intrinsically brighter (more negative M)."""
    return slope * math.log10(period_days) + zero_point


def cepheid_distance(period_days: float, m_app: float, slope: float = -2.81,
                     zero_point: float = -1.43) -> float:
    """Distance (pc) to a Cepheid from its period and apparent magnitude, via the
    P-L relation and the distance modulus."""
    M = cepheid_absolute_magnitude(period_days, slope, zero_point)
    return distance_from_modulus(m_app - M)
