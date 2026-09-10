"""Binary-pulsar timing: the Hulse-Taylor orbital decay (first proof of GW).

PSR B1913+16 is a pulsar in a tight, eccentric orbit with another neutron star.
Its pulses are a clock precise enough to track the orbit, and that orbit is
SHRINKING: the system loses energy to gravitational waves exactly as general
relativity predicts. Watching the orbital period decay was the first
(indirect) evidence for gravitational waves -- the 1993 Nobel Prize, two decades
before LIGO's direct detection.

The orbit-averaged period-decay rate is the Peters (1964) result:

    dP/dt = - (192 pi / 5) (2 pi G / (P c^3))^{5/3}
              * (m1 m2) / (m1 + m2)^{1/3}
              * (1 + 73/24 e^2 + 37/96 e^4) / (1 - e^2)^{7/2}.

For B1913+16's measured masses, period, and eccentricity this gives
dP/dt ~ -2.4e-12 (s/s), matching the observed value to ~0.2%. The cumulative
shift in periastron time is a parabola in elapsed time -- the famous plot whose
points fall on the GR curve.

SI units. Pure stdlib.
"""

from __future__ import annotations

import math

G = 6.67430e-11
C = 2.99792458e8
M_SUN = 1.98892e30
YEAR = 3.15576e7

# PSR B1913+16 measured parameters (Weisberg & Huang 2016 / classic values)
HT_P_ORB = 27906.98           # orbital period, seconds (~7.75 hr)
HT_E = 0.6171334              # eccentricity
HT_M1 = 1.438 * M_SUN         # pulsar mass
HT_M2 = 1.390 * M_SUN         # companion mass


def period_decay_rate(P: float, e: float, m1: float, m2: float) -> float:
    """Orbit-averaged dP/dt (dimensionless, s/s) from gravitational-wave
    emission -- the Peters formula for the orbital period."""
    M = m1 + m2
    ecc_factor = (1.0 + (73.0 / 24.0) * e * e + (37.0 / 96.0) * e ** 4) \
        / (1.0 - e * e) ** 3.5
    pref = -(192.0 * math.pi / 5.0) * (2.0 * math.pi * G / (P * C ** 3)) ** (5.0 / 3.0)
    return pref * (m1 * m2) / M ** (1.0 / 3.0) * ecc_factor


def hulse_taylor_pdot() -> float:
    """dP/dt for PSR B1913+16 from its measured parameters."""
    return period_decay_rate(HT_P_ORB, HT_E, HT_M1, HT_M2)


def cumulative_periastron_shift(t: float, P: float = HT_P_ORB, e: float = HT_E,
                                m1: float = HT_M1, m2: float = HT_M2) -> float:
    """Cumulative shift (seconds) in the time of periastron after elapsed time t,
    the integral of the accumulating period change: for a slowly changing period,
    Delta = (1/2) (Pdot / P) t^2. This parabola is the Hulse-Taylor plot."""
    pdot = period_decay_rate(P, e, m1, m2)
    return 0.5 * (pdot / P) * t * t


def merger_time(P: float = HT_P_ORB, e: float = HT_E,
                m1: float = HT_M1, m2: float = HT_M2) -> float:
    """Rough time until merger, P / |dP/dt| scaled -- actually the orbit
    decays faster as it tightens, but the current e-folding P/|Pdot| gives the
    order of magnitude (~300 Myr for B1913+16)."""
    pdot = period_decay_rate(P, e, m1, m2)
    return P / abs(pdot)
