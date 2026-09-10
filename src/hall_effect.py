"""The Hall effect: reading a conductor's charge carriers with a magnet.

Run a current through a conductor and put it in a perpendicular magnetic field. The moving
charges are pushed sideways by the Lorentz force and pile up on one edge, building a
transverse voltage that grows until its electric field just balances the magnetic push. That
Hall voltage is

    V_H = I B / (n q t),

for current I, field B, carrier density n, charge q, and sample thickness t. Measuring it
does three things at once: it gives the carrier density n, and its *sign* tells you whether
the charge carriers are electrons or holes -- the discovery that some metals and many
semiconductors conduct with positive holes, which classical physics could not explain and
which underpins all semiconductor doping.

The Hall coefficient R_H = 1 / (n q) = E_y / (j_x B) packages this, and combined with the
conductivity it separates carrier density from mobility:

    mu = |R_H| sigma,

so a Hall-bar measurement plus a resistance measurement fully characterizes a conductor. The
Hall angle theta_H = arctan(mu B) measures how far the field tilts the current, and at high
field and low temperature the effect becomes quantized (the von Klitzing resistance
h/e^2 ~ 25.8 kOhM), but this module treats the classical Drude Hall effect.

This module gives the Hall voltage, coefficient, carrier density, mobility, Hall angle and
carrier sign, and reproduces copper's tiny electron Hall voltage and the density inferred
from it. SI units. Pure stdlib; the transport companion to the Drude/cyclotron and
uncertainty notes.
"""

from __future__ import annotations

import math

E_CHARGE = 1.602176634e-19    # elementary charge (C)


def hall_voltage(current: float, field: float, carrier_density: float, thickness: float,
                 charge: float = E_CHARGE) -> float:
    """Hall voltage V_H = I B / (n q t) (V) across a sample of thickness t carrying current
    I in a perpendicular field B, with carrier density n and carrier charge q."""
    return current * field / (carrier_density * charge * thickness)


def hall_coefficient(carrier_density: float, charge: float = E_CHARGE) -> float:
    """Hall coefficient R_H = 1 / (n q) (m^3/C). Negative for electron conduction (q = -e),
    positive for holes -- its sign identifies the carrier."""
    return 1.0 / (carrier_density * charge)


def carrier_density(current: float, field: float, hall_v: float, thickness: float,
                    charge: float = E_CHARGE) -> float:
    """Carrier density n = I B / (V_H q t) (1/m^3) inferred from a measured Hall voltage.
    Inverts hall_voltage."""
    return current * field / (hall_v * charge * thickness)


def hall_mobility(hall_coeff: float, conductivity: float) -> float:
    """Carrier mobility mu = |R_H| sigma (m^2/(V s)) from the Hall coefficient and the
    conductivity -- how a Hall bar plus a resistance measurement separate density from
    mobility."""
    return abs(hall_coeff) * conductivity


def hall_angle(mobility: float, field: float) -> float:
    """Hall angle theta_H = arctan(mu B) (rad): the tilt of the current from the applied
    field direction. Small in a poor conductor, approaching 90 deg for high mobility x field."""
    return math.atan(mobility * field)


def carrier_sign(hall_coeff: float) -> str:
    """'electrons' if the Hall coefficient is negative, 'holes' if positive. The Hall effect's
    headline result: it reveals the sign of the mobile charge."""
    return "holes" if hall_coeff > 0 else "electrons"
