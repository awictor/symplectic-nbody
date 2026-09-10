"""The Froude number: racing your own waves.

A disturbance on a water surface travels as a gravity wave; in shallow water of depth h that
speed is c = sqrt(g h). Compare the flow (or a boat's) speed U to it and you get the Froude
number,

    Fr = U / sqrt(g h),

the single most important number in free-surface and ship hydrodynamics. Fr < 1 is
*subcritical* (tranquil) flow: disturbances outrun the current and travel upstream, so the
surface can adjust ahead of an obstacle. Fr > 1 is *supercritical* (shooting) flow: the water
moves faster than its own waves, nothing propagates upstream, and a sudden slowing produces a
hydraulic jump -- the abrupt, turbulent step you see below a spillway or around a kitchen-sink
disc of fast water. Fr = 1 is critical flow, the condition at the crest of a weir.

For a ship the relevant length is the waterline length L, and the *hull* Froude number
Fr = U/sqrt(g L) governs wave-making drag. A displacement hull hits a wall near Fr ~ 0.4:
its bow and stern waves reinforce and it must climb its own bow wave, the origin of the
"hull speed" rule V_hull ~ 1.34 sqrt(L_ft) knots. The Kelvin ship-wake wedge sits at a fixed
half-angle of 19.47 degrees regardless of speed, one of the prettiest results in the theory.

Across a hydraulic jump the depths obey the Belanger conjugate relation, and momentum
conservation fixes the downstream depth from the upstream Froude number. This module gives
the shallow-water wave speed, the flow and hull Froude numbers, the flow regime, the critical
depth, the hull speed, the conjugate depth of a hydraulic jump, and the Kelvin wake angle,
and reproduces the Fr ~ 0.4 hull-speed wall and the 19.47-degree wedge. SI units. Pure stdlib;
the free-surface companion to the capillary and Bernoulli notes.
"""

from __future__ import annotations

import math

G_EARTH = 9.80665
KELVIN_HALF_ANGLE_DEG = math.degrees(math.asin(1.0 / 3.0))   # 19.4712...


def wave_speed(depth: float, g: float = G_EARTH) -> float:
    """Shallow-water gravity-wave speed c = sqrt(g h) (m/s): how fast a surface disturbance
    of a long wavelength travels in water of depth h."""
    return math.sqrt(g * depth)


def froude_number(velocity: float, depth: float, g: float = G_EARTH) -> float:
    """Flow Froude number Fr = U / sqrt(g h): flow speed over shallow-water wave speed.
    <1 subcritical (tranquil), =1 critical, >1 supercritical (shooting)."""
    return velocity / wave_speed(depth, g)


def hull_froude(velocity: float, waterline_length: float, g: float = G_EARTH) -> float:
    """Hull Froude number Fr = U / sqrt(g L) for a ship of waterline length L: governs
    wave-making resistance. The classic hull-speed wall sits near Fr ~ 0.4."""
    return velocity / math.sqrt(g * waterline_length)


def flow_regime(froude: float) -> str:
    """Classify a flow by its Froude number: 'subcritical' (<1, tranquil, waves go upstream),
    'critical' (=1), or 'supercritical' (>1, shooting, nothing propagates upstream)."""
    if abs(froude - 1.0) < 1e-9:
        return "critical"
    return "subcritical" if froude < 1.0 else "supercritical"


def critical_depth(flow_per_width: float, g: float = G_EARTH) -> float:
    """Critical depth h_c = (q^2 / g)^(1/3) (m) for a specific discharge q (flow per unit
    width, m^2/s): the depth at which Fr = 1 and specific energy is minimized."""
    return (flow_per_width * flow_per_width / g) ** (1.0 / 3.0)


def hull_speed(waterline_length: float, g: float = G_EARTH) -> float:
    """Displacement hull speed (m/s) at the Fr ~ 0.4028 wave-making wall:
    V = 0.4028 sqrt(g L). Equivalent to the 1.34 sqrt(L_ft) knots rule of thumb."""
    return 0.4028 * math.sqrt(g * waterline_length)


def conjugate_depth(upstream_depth: float, upstream_froude: float) -> float:
    """Downstream (sequent) depth of a hydraulic jump from the upstream depth and Froude
    number, Belanger's relation h2/h1 = 1/2 (sqrt(1 + 8 Fr1^2) - 1). Only a supercritical
    upstream flow (Fr1 > 1) produces a real jump to a deeper, slower stream."""
    ratio = 0.5 * (math.sqrt(1.0 + 8.0 * upstream_froude ** 2) - 1.0)
    return upstream_depth * ratio


def kelvin_wake_half_angle(g: float = G_EARTH) -> float:
    """Half-angle (degrees) of the Kelvin ship-wake wedge: arcsin(1/3) ~ 19.47 deg,
    independent of the ship's speed (in deep water)."""
    return KELVIN_HALF_ANGLE_DEG
