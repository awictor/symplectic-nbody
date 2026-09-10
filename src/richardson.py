"""The Richardson number: when shear beats stratification into turbulence.

A stably stratified fluid -- warm light air over cold dense air, or fresh water over salt --
resists overturning: lift a parcel and buoyancy pushes it back, oscillating at the Brunt-
Vaisala frequency N. But if the layers slide past each other fast enough, the shear can rip
the interface into billows and mix it anyway. Which wins is the gradient Richardson number,

    Ri = N^2 / (du/dz)^2,

the buoyant restoring stiffness N^2 over the square of the velocity shear du/dz. Large Ri
(strong stratification, weak shear) stays laminar and layered; small Ri lets shear
instabilities grow. The Miles-Howard theorem gives a sharp sufficient condition for
stability: if Ri > 1/4 everywhere, the flow cannot go unstable to this mechanism. Below
Ri = 1/4 the Kelvin-Helmholtz instability can grow, curling the interface into the row of
"cat's-eye" billows you see in cloud edges and on the surface of a slow river.

Ri governs clear-air turbulence that jolts aircraft, the mixing (or lack of it) in the ocean
thermocline and atmospheric inversions, and the entrainment at the top of a fog layer. The
bulk Richardson number Ri_b = g (drho/rho) L / U^2 is the same idea from finite differences
across a layer of depth L.

This module gives the gradient and bulk Richardson numbers, the Kelvin-Helmholtz stability
test, the critical shear that marginally destabilizes a given stratification, and the
Brunt-Vaisala frequency it builds on, and reproduces the Ri = 1/4 threshold. SI units. Pure
stdlib; the stratified-turbulence companion to the Brunt-Vaisala and Reynolds notes.
"""

from __future__ import annotations

import math

G_EARTH = 9.80665
RI_CRITICAL = 0.25             # Miles-Howard critical Richardson number


def gradient_richardson(n_squared: float, shear: float) -> float:
    """Gradient Richardson number Ri = N^2 / (du/dz)^2: buoyant stiffness over squared
    velocity shear. Large = stable/laminar, < 1/4 = shear instability possible."""
    return n_squared / (shear * shear)


def bulk_richardson(delta_rho: float, rho: float, length: float, delta_u: float,
                    g: float = G_EARTH) -> float:
    """Bulk Richardson number Ri_b = g (drho/rho) L / (du)^2 across a layer of depth L with
    density jump drho and velocity jump du. The finite-difference form of Ri."""
    return g * (delta_rho / rho) * length / (delta_u * delta_u)


def is_kh_stable(ri: float, ri_critical: float = RI_CRITICAL) -> bool:
    """True if the flow is stable to the Kelvin-Helmholtz mechanism (Ri > 1/4, Miles-Howard
    sufficient condition). Below 1/4 shear billows can grow."""
    return ri > ri_critical


def brunt_vaisala_frequency(n_squared: float) -> float:
    """Brunt-Vaisala (buoyancy) frequency N = sqrt(N^2) (rad/s) for a stable layer (N^2>0);
    returns 0 for a neutral or unstable layer."""
    return math.sqrt(n_squared) if n_squared > 0.0 else 0.0


def critical_shear(n_squared: float, ri_critical: float = RI_CRITICAL) -> float:
    """Velocity shear du/dz (1/s) at which the gradient Richardson number falls to the
    critical 1/4, marginally destabilizing the stratification: shear = sqrt(N^2 / Ri_c).
    Steeper shear than this can trigger Kelvin-Helmholtz billows."""
    return math.sqrt(n_squared / ri_critical)


def n_squared_from_density(drho_dz: float, rho: float, g: float = G_EARTH) -> float:
    """Brunt-Vaisala N^2 = -(g/rho) drho/dz (1/s^2) from a density gradient. Positive
    (stable) when density decreases upward (drho/dz < 0)."""
    return -(g / rho) * drho_dz
