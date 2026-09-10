"""The curve of growth: reading column densities from absorption-line strength.

The equivalent width W of an absorption line -- the width of a fully-black rectangle
removing the same flux -- measures how much light the line takes out. How W grows with
the column density N of absorbing atoms follows three distinct regimes, together the
"curve of growth," one of the workhorses of quantitative spectroscopy.

1. LINEAR (weak lines). At low column density the line is optically thin and every
   extra atom removes proportionally more flux:

       W ~ N.

2. FLAT (saturated). Once the line centre goes optically thick it is already black, so
   adding atoms barely widens it -- W creeps up only as sqrt(ln N). Column density is
   hard to measure here; the line is "saturated."

3. SQUARE-ROOT (damped). At very high column density the Lorentzian damping wings
   (natural + pressure broadening) become optically thick and the line grows again,

       W ~ sqrt(N).

The transitions are set by the central optical depth tau0 ~ N and the ratio of the
Lorentzian damping width to the Doppler width. This module gives tau0, the equivalent
width across all three regimes, and the regime name, and reproduces the linear ->
saturated -> damped progression. Dimensionless / consistent units. Pure stdlib; the
column-density companion to the line-broadening and Saha modules.
"""

from __future__ import annotations

import math


def central_optical_depth(N: float, f: float, doppler_width: float,
                          cross_section_const: float = 1.0) -> float:
    """Central optical depth tau0 of a line, proportional to column density N times
    oscillator strength f divided by the Doppler width. tau0 ~ N f / delta_nu_D."""
    return cross_section_const * N * f / doppler_width


def equivalent_width(tau0: float, damping_ratio: float = 1e-3) -> float:
    """Equivalent width W (in Doppler-width units) as a function of central optical
    depth tau0, spanning the three regimes:

      linear     tau0 << 1:   W ~ tau0
      saturated  tau0 >~ 1:   W ~ sqrt(ln tau0)
      damped     tau0 huge:   W ~ sqrt(tau0 * damping_ratio)

    damping_ratio = (Lorentzian width / Doppler width) sets where damping wings take
    over. Returns W / delta_nu_D (dimensionless)."""
    if tau0 <= 0.0:
        return 0.0
    linear = tau0
    saturated = 2.0 * math.sqrt(math.log(1.0 + tau0))
    damped = math.sqrt(math.pi * tau0 * damping_ratio)
    # the three branches cross over in order; the observed W is the smallest that
    # still exceeds the previous regime -- captured by taking the min of the linear
    # branch and the saturated branch (saturation caps the linear growth), then the
    # max with the damped wings that revive growth at very high tau0.
    return max(min(linear, saturated), damped)


def regime(tau0: float, damping_ratio: float = 1e-3) -> str:
    """Name the curve-of-growth regime for a given central optical depth:
    'linear', 'saturated', or 'damped'."""
    if tau0 < 1.0:
        return "linear"
    # damping wings dominate once sqrt(pi tau0 a) exceeds the saturated value
    saturated = math.sqrt(math.log(1.0 + tau0)) * 2.0
    damped = math.sqrt(math.pi * tau0 * damping_ratio)
    return "damped" if damped > saturated else "saturated"


def column_from_linear_width(W: float, f: float, doppler_width: float,
                             cross_section_const: float = 1.0) -> float:
    """Invert the linear regime to recover column density N from a weak line's
    equivalent width: N = W delta_nu_D / (const f). Valid only where W ~ N."""
    return W * doppler_width / (cross_section_const * f)
