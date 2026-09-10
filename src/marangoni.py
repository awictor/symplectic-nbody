"""The Marangoni effect: flow driven by a gradient in surface tension.

Surface tension usually just pulls a surface taut, but when it *varies* along the surface --
because temperature or composition changes from place to place -- the imbalance drags the
fluid itself. Liquid is pulled from regions of low surface tension toward regions of high
surface tension, and it hauls the underlying fluid with it. This is the Marangoni effect, and
it explains the "tears of wine" that climb a glass, the way a drop of soap sends pepper
flakes fleeing across water, and thermocapillary convection in a weld pool or a microgravity
experiment where buoyancy is absent.

The strength of the effect relative to viscous damping is the Marangoni number,

    Ma = (dgamma/dT) dT L / (mu alpha),

for a thermal gradient (dgamma/dT is how surface tension changes with temperature, usually
negative; L a length, mu the dynamic viscosity, alpha the thermal diffusivity). For a
solutal gradient replace dT with the concentration difference and alpha with the mass
diffusivity D. Above a critical Ma ~ 80 a heated liquid layer breaks into steady
Benard-Marangoni convection cells (the surface-tension-driven cousin of Rayleigh-Benard).

Which mechanism dominates a heated layer -- surface tension or buoyancy -- is set by the
dynamic Bond number Bo_d = Ra/Ma = rho g beta L^2 / (dgamma/dT): thin layers and low gravity
are Marangoni-driven, thick layers on the ground are buoyancy-driven. This module gives the
thermal and solutal Marangoni numbers, the onset test, the dynamic Bond number, and the
characteristic Marangoni flow speed, and reproduces the tears-of-wine surface stress and the
Ma ~ 80 onset. SI units. Pure stdlib; the interface-flow companion to the surface-tension and
Rayleigh-Benard notes.
"""

from __future__ import annotations

MA_CRITICAL = 80.0             # onset of Benard-Marangoni convection (free-slip ~ 80)


def marangoni_thermal(dgamma_dT: float, delta_T: float, length: float,
                      mu: float, alpha: float) -> float:
    """Thermal Marangoni number Ma = |dgamma/dT| dT L / (mu alpha): surface-tension drive
    from a temperature gradient over viscous + thermal diffusion. dgamma_dT may be given
    signed; its magnitude sets the drive."""
    return abs(dgamma_dT) * delta_T * length / (mu * alpha)


def marangoni_solutal(dgamma_dc: float, delta_c: float, length: float,
                      mu: float, diffusivity: float) -> float:
    """Solutal Marangoni number from a concentration gradient: |dgamma/dc| dc L / (mu D).
    Drives flows like tears of wine (ethanol evaporation raises local surface tension)."""
    return abs(dgamma_dc) * delta_c * length / (mu * diffusivity)


def surface_stress(dgamma_dT: float, dT_dx: float) -> float:
    """Marangoni shear stress at the surface tau = dgamma/dT * dT/dx (Pa): the tangential
    pull per area from a surface-tension gradient. Sets the interfacial flow."""
    return dgamma_dT * dT_dx


def is_convecting(ma: float, ma_critical: float = MA_CRITICAL) -> bool:
    """True if the Marangoni number exceeds the critical value (~80) so a heated layer breaks
    into steady Benard-Marangoni convection cells."""
    return ma > ma_critical


def dynamic_bond_number(rho: float, g: float, beta: float, length: float,
                        dgamma_dT: float) -> float:
    """Dynamic Bond number Bo_d = Ra/Ma = rho g beta L^2 / |dgamma/dT|: buoyancy vs surface-
    tension drive in a heated layer. >>1 buoyancy-dominated (thick/ground), <<1 Marangoni-
    dominated (thin/microgravity)."""
    return rho * g * beta * length * length / abs(dgamma_dT)


def marangoni_velocity(dgamma_dT: float, delta_T: float, length: float, mu: float) -> float:
    """Characteristic Marangoni flow speed U ~ |dgamma/dT| dT / mu (m/s): the surface-tension
    stress balanced against viscous shear over the layer (the L's cancel in the scaling)."""
    return abs(dgamma_dT) * delta_T / mu
