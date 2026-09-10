"""The CMB acoustic scale: why the microwave-background spots are ~1 degree.

Before recombination the universe is a photon-baryon plasma in which sound waves
propagate. The farthest a wave can travel by recombination is the SOUND HORIZON

    r_s = integral_0^{t_rec} c_s (1+z) dt  ~  c_s * (age at recombination),

with sound speed c_s = c / sqrt(3(1+R)), R the baryon-to-photon momentum ratio.
That fixed length, seen across the vast distance to the last-scattering surface,
subtends an angle

    theta = r_s / D_A,

which shows up as the first peak of the CMB power spectrum at multipole

    l ~ pi / theta  ~  200,

i.e. features about 1 degree across. Measuring that peak (l ~ 220, WMAP/Planck)
pins the geometry of the universe to be flat. This module estimates the sound
horizon, the angular scale, and the peak multipole, reusing the recombination
redshift (saha) and the angular-diameter distance (distances/friedmann).

Simplified matter-dominated estimates; the numbers land in the right place.
SI-ish, distances in Mpc. Pure stdlib.
"""

from __future__ import annotations

import math

from friedmann import Cosmology
from distances import comoving_distance
from saha import recombination_redshift

C_KM_S = 299792.458
MPC = 1.0                       # work in Mpc; c/H0 sets the scale


def sound_speed_fraction(R: float) -> float:
    """Photon-baryon sound speed as a fraction of c: c_s/c = 1/sqrt(3(1+R))."""
    return 1.0 / math.sqrt(3.0 * (1.0 + R))


def sound_horizon(cosmo: Cosmology, z_rec: float, H0: float = 70.0,
                  R_rec: float = 0.6) -> float:
    """Comoving sound horizon at recombination (Mpc). Approximates the integral
    of c_s / (a^2 H) from the Big Bang to z_rec in a matter+radiation universe:

        r_s ~ (2 c / (H0 sqrt(3 Omega_m))) * (1/sqrt(1+z_rec)) / sqrt(1+R_rec).

    This is the standard matter-dominated closed form (Hu & Sugiyama), good to
    ~10%."""
    hubble_dist = C_KM_S / H0                       # c/H0 in Mpc
    cs_frac = 1.0 / math.sqrt(1.0 + R_rec)
    return (2.0 * hubble_dist / math.sqrt(3.0 * cosmo.Omega_m)) \
        * (1.0 / math.sqrt(1.0 + z_rec)) * cs_frac


def angular_diameter_distance_to_rec(cosmo: Cosmology, z_rec: float,
                                     H0: float = 70.0) -> float:
    """Angular-diameter distance to the last-scattering surface (Mpc):
    D_A = D_C / (1+z_rec), with D_C the comoving distance in Mpc."""
    hubble_dist = C_KM_S / H0
    D_C = comoving_distance(cosmo, z_rec) * hubble_dist
    return D_C / (1.0 + z_rec)


def acoustic_angle(cosmo: Cosmology, z_rec: float, H0: float = 70.0,
                   R_rec: float = 0.6) -> float:
    """Angular size (radians) of the sound horizon: theta = r_s / D_A.
    Uses the comoving sound horizon and comoving distance (both comoving)."""
    r_s = sound_horizon(cosmo, z_rec, H0, R_rec)
    hubble_dist = C_KM_S / H0
    D_C = comoving_distance(cosmo, z_rec) * hubble_dist
    return r_s / D_C


def first_peak_multipole(cosmo: Cosmology, z_rec: float = None,
                         H0: float = 70.0, R_rec: float = 0.6) -> float:
    """First acoustic-peak multipole l ~ pi / theta."""
    if z_rec is None:
        z_rec = recombination_redshift(0.5)
    return math.pi / acoustic_angle(cosmo, z_rec, H0, R_rec)
