"""The Sunyaev-Zeldovich effect: galaxy clusters shadowing the CMB.

Hot electrons in a galaxy cluster's gas inverse-Compton scatter passing cosmic
microwave background photons, nudging them up in energy. The cluster imprints a
tiny, characteristic spectral distortion on the CMB -- the thermal
Sunyaev-Zeldovich (SZ) effect -- set by the Compton y-parameter,

    y = integral (k_B T_e / m_e c^2) sigma_T n_e dl,

the pressure of the electron gas along the line of sight in natural units. In the
Rayleigh-Jeans (low-frequency) part of the spectrum the cluster appears as a
DECREMENT,

    Delta T / T_CMB = -2 y,

while at high frequency it is an increment, with a null near 217 GHz. For a
massive cluster y ~ 1e-4. Crucially the SZ signal is REDSHIFT-INDEPENDENT (it is
a fractional distortion of the CMB), so it finds clusters at any distance --
which is why SZ surveys are a premier cluster-cosmology probe.

This module gives the y-parameter, the RJ temperature decrement, and reproduces
the cluster-scale values. SI units. Pure stdlib; complements the cluster and
inverse-Compton modules.
"""

from __future__ import annotations

import math

K_B = 1.380649e-23
M_E = 9.1093837e-31
C = 2.99792458e8
SIGMA_T = 6.6524587e-29
KEV = 1.602176634e-16
MPC = 3.0857e22
T_CMB = 2.725                  # K


def y_parameter(n_e: float, T_e: float, path_length: float) -> float:
    """Compton y for a uniform slab: y = (k_B T_e / m_e c^2) sigma_T n_e L."""
    return (K_B * T_e / (M_E * C ** 2)) * SIGMA_T * n_e * path_length


def y_from_kev(n_e: float, kT_keV: float, path_length: float) -> float:
    """y-parameter with the electron temperature given as kT in keV."""
    kT_joule = kT_keV * KEV
    return (kT_joule / (M_E * C ** 2)) * SIGMA_T * n_e * path_length


def rj_temperature_decrement(y: float) -> float:
    """Fractional CMB temperature change in the Rayleigh-Jeans limit: -2 y."""
    return -2.0 * y


def rj_decrement_kelvin(y: float) -> float:
    """Absolute RJ temperature decrement in kelvin: Delta T = -2 y T_CMB."""
    return rj_temperature_decrement(y) * T_CMB


def is_redshift_independent() -> bool:
    """The SZ distortion is a fractional change to the CMB spectrum itself, so it
    does not dim with distance -- unlike surface brightness, which falls as
    (1+z)^-4. This flag documents that property (always True)."""
    return True
