"""The Strouhal number: the rhythm of a von Karman vortex street.

Put a blunt body -- a cylinder, a chimney, a wire, a tree trunk -- in a steady flow and it
does not just shed a smooth wake. Above a modest Reynolds number the wake becomes unstable
and sheds vortices alternately from each side, a staggered double row called a von Karman
vortex street. The shedding is astonishingly regular, and its frequency f obeys a
dimensionless law, the Strouhal number:

    St = f d / U,

for a body of cross-stream size d in a flow of speed U. Over an enormous range of Reynolds
number (roughly 300 to 2e5) a circular cylinder sheds at St ~ 0.2 -- nearly constant -- so
the shedding frequency scales linearly with wind speed, f = St U / d. That single fact
explains a lot: the singing of telephone wires and rigging (aeolian tones), the fluttering
hum of a car antenna, and the resonant sway that can destroy a chimney or a bridge deck when
the shedding frequency crosses a structural natural frequency (vortex-induced vibration, or
lock-in).

Roshko refined the constant Reynolds-number dependence to St = 0.212 (1 - 21.2/Re) for the
cylinder, capturing the slow rise of St toward ~0.21 as Re grows. This module gives the
shedding frequency, the Strouhal number, the Roshko relation, the aeolian-tone pitch, the
flow speed that would lock the shedding onto a structure's natural frequency, and the vortex
spacing, and reproduces the St ~ 0.2 plateau and the audible pitch of wind over a wire. SI
units. Pure stdlib; the wake-dynamics companion to the Reynolds and Blasius notes.
"""

from __future__ import annotations

ST_CYLINDER = 0.2              # canonical Strouhal number for a circular cylinder


def shedding_frequency(velocity: float, diameter: float,
                       strouhal: float = ST_CYLINDER) -> float:
    """Vortex-shedding frequency f = St U / d (Hz) for a body of size d in a flow of speed U."""
    return strouhal * velocity / diameter


def strouhal_number(frequency: float, diameter: float, velocity: float) -> float:
    """Strouhal number St = f d / U from a measured shedding frequency."""
    return frequency * diameter / velocity


def roshko_strouhal(reynolds: float) -> float:
    """Roshko's fit for a circular cylinder, St = 0.212 (1 - 21.2/Re): the Strouhal number
    rises with Reynolds number toward ~0.21. Valid for roughly Re > 300."""
    return 0.212 * (1.0 - 21.2 / reynolds)


def aeolian_frequency(velocity: float, diameter: float,
                      strouhal: float = ST_CYLINDER) -> float:
    """Pitch (Hz) of the aeolian tone -- the audible hum of wind over a wire or string, which
    is just the vortex-shedding frequency St U / d."""
    return shedding_frequency(velocity, diameter, strouhal)


def lock_in_velocity(natural_frequency: float, diameter: float,
                     strouhal: float = ST_CYLINDER) -> float:
    """Flow speed (m/s) at which the shedding frequency equals a structure's natural
    frequency, U = f_n d / St -- the wind speed that risks resonant vortex-induced vibration
    (lock-in)."""
    return natural_frequency * diameter / strouhal


def vortex_spacing(velocity: float, diameter: float,
                   strouhal: float = ST_CYLINDER) -> float:
    """Streamwise distance between successive same-side vortices, the wake wavelength
    lambda = U / f = d / St (m). For St ~ 0.2 the vortices sit ~5 diameters apart."""
    return velocity / shedding_frequency(velocity, diameter, strouhal)
