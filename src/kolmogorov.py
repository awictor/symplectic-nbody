"""The Kolmogorov cascade: how turbulence shreds big eddies into heat.

Stir a fluid hard and you make big, energetic eddies. They are unstable and break into
smaller ones, which break into smaller ones still, handing their energy down a cascade until
the eddies are so tiny that viscosity finally smears them into heat. Kolmogorov's 1941 theory
(K41) says that in between -- the inertial range -- the statistics depend only on the rate
epsilon at which energy is passed down (equal to the rate it is fed in and dissipated) and on
the eddy size, giving by dimensional analysis the famous energy spectrum

    E(k) = C epsilon^(2/3) k^(-5/3),      C ~ 1.5,

the -5/3 power law confirmed from wind tunnels to the ocean to interstellar gas. The cascade
runs from the large stirring scale L down to the Kolmogorov dissipation scale where inertia
and viscosity balance:

    eta = (nu^3 / epsilon)^(1/4)     (length)
    tau_eta = (nu / epsilon)^(1/2)   (time)
    u_eta = (nu epsilon)^(1/4)       (velocity),   all with a Reynolds number of order 1.

The span of the cascade is set by the large-scale Reynolds number: L/eta ~ Re^(3/4), so the
number of eddy scales -- and hence the cost of resolving turbulence on a computer -- explodes
as Re^(9/4) in 3-D. This module gives the dissipation rate from the large scales, the
Kolmogorov length/time/velocity, the inertial-range spectrum and eddy turnover time, and the
scale-separation L/eta, and reproduces the -5/3 slope and the Re^(3/4) range. SI units. Pure
stdlib; the turbulence companion to the Reynolds and Richardson notes.
"""

from __future__ import annotations

KOLMOGOROV_CONSTANT = 1.5      # C in E(k) = C epsilon^(2/3) k^(-5/3)


def dissipation_rate(u_rms: float, length: float) -> float:
    """Energy dissipation rate epsilon ~ u^3 / L (W/kg): at steady state the big eddies feed
    energy down the cascade at the rate their own turnover sets, independent of viscosity."""
    return u_rms ** 3 / length


def kolmogorov_length(nu: float, epsilon: float) -> float:
    """Kolmogorov dissipation length eta = (nu^3 / epsilon)^(1/4) (m): the smallest eddy,
    where viscosity finally wins and motion turns to heat."""
    return (nu ** 3 / epsilon) ** 0.25


def kolmogorov_time(nu: float, epsilon: float) -> float:
    """Kolmogorov time tau_eta = (nu / epsilon)^(1/2) (s): the turnover time of the smallest
    eddies."""
    return (nu / epsilon) ** 0.5


def kolmogorov_velocity(nu: float, epsilon: float) -> float:
    """Kolmogorov velocity u_eta = (nu epsilon)^(1/4) (m/s): the speed of the smallest eddies;
    their Reynolds number u_eta eta / nu is exactly 1."""
    return (nu * epsilon) ** 0.25


def energy_spectrum(k: float, epsilon: float, c: float = KOLMOGOROV_CONSTANT) -> float:
    """Inertial-range energy spectrum E(k) = C epsilon^(2/3) k^(-5/3) (m^3/s^2): energy per
    unit wavenumber, the -5/3 law."""
    return c * epsilon ** (2.0 / 3.0) * k ** (-5.0 / 3.0)


def eddy_turnover_time(size: float, epsilon: float) -> float:
    """Turnover time of an eddy of size l in the inertial range, tau(l) = (l^2/epsilon)^(1/3)
    (s): smaller eddies turn over faster, so the cascade accelerates downward."""
    return (size * size / epsilon) ** (1.0 / 3.0)


def scale_separation(reynolds: float) -> float:
    """Ratio of the largest to smallest eddy, L/eta ~ Re^(3/4): the width of the inertial
    range. It sets why resolving turbulence costs ~Re^(9/4) grid points in 3-D."""
    return reynolds ** 0.75
