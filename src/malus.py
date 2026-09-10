"""Malus's law: turning light down with a twist of a polarizer.

Light is a transverse wave, and its electric field can point in any direction perpendicular
to travel -- its polarization. A polarizer passes only the field component along its
transmission axis, so linearly polarized light of intensity I0 hitting a polarizer at angle
theta to that axis comes out at

    I = I0 cos^2(theta),

Malus's law (1809). Two consequences run through all of optics. First, unpolarized light --
an even mix of all angles -- loses exactly half its intensity through any single polarizer,
because <cos^2> = 1/2. Second, two polarizers crossed at 90 degrees pass nothing (cos^2 90 =
0), the "crossed polarizers" that make an LCD pixel dark; but slip a third polarizer between
them at 45 degrees and light reappears, since each 45-degree step only costs a factor of two.

A stack of N polarizers each rotated by a small angle theta/N can rotate polarization through
theta while passing (cos^2(theta/N))^N -> 1 as N grows -- an optical version of the quantum
Zeno effect. Wave plates do the rotation without loss by retarding one component: a
half-wave plate (retardance pi) reflects the polarization angle, a quarter-wave plate
(retardance pi/2) turns linear light circular.

This module gives the Malus transmission, the unpolarized-through-one-polarizer half, the
throughput of a polarizer stack (including the three-polarizer trick), the extinction ratio
of crossed polarizers, and the wave-plate retardance, and reproduces the cos^2 law and the
45-degree rescue. SI units, angles in radians unless _deg. Pure stdlib; the wave-optics
companion to the Snell and thin-film notes.
"""

from __future__ import annotations

import math


def malus_transmission(i0: float, theta: float) -> float:
    """Transmitted intensity I = I0 cos^2(theta) through a polarizer at angle theta to the
    incoming linear polarization (Malus's law)."""
    c = math.cos(theta)
    return i0 * c * c


def unpolarized_through_one(i0: float) -> float:
    """Intensity of unpolarized light after one polarizer: I0/2, since the average of
    cos^2 over all angles is 1/2. The light is then linearly polarized along the axis."""
    return i0 / 2.0


def two_polarizer_transmission(i0_unpolarized: float, angle_between: float) -> float:
    """Unpolarized light through two polarizers whose axes differ by angle_between:
    (I0/2) cos^2(angle). Zero at 90 degrees (crossed)."""
    return unpolarized_through_one(i0_unpolarized) * math.cos(angle_between) ** 2


def stack_transmission(i0: float, total_angle: float, n_polarizers: int) -> float:
    """Linearly polarized light through a stack of N polarizers, each rotated by
    total_angle/N from the previous: I0 (cos^2(total_angle/N))^N. Rotates the polarization by
    total_angle; approaches I0 as N grows (quantum-Zeno-like)."""
    step = total_angle / n_polarizers
    return i0 * (math.cos(step) ** 2) ** n_polarizers


def three_polarizer_rescue(i0_unpolarized: float, middle_angle: float = math.pi / 4.0) -> float:
    """Unpolarized light through crossed (0 and 90 deg) polarizers with a third inserted at
    middle_angle: (I0/2) cos^2(theta) cos^2(90deg - theta). Nonzero for 0<theta<90 even
    though the outer pair alone passes nothing."""
    i_after_first = unpolarized_through_one(i0_unpolarized)
    i_after_mid = i_after_first * math.cos(middle_angle) ** 2
    return i_after_mid * math.cos(math.pi / 2.0 - middle_angle) ** 2


def extinction_ratio(theta_misalignment: float) -> float:
    """Fraction of light leaking through nominally crossed polarizers misaligned from exactly
    90 degrees by theta_misalignment: cos^2(90deg - theta) = sin^2(theta). Zero when perfectly
    crossed."""
    return math.sin(theta_misalignment) ** 2


def waveplate_retardance(thickness: float, birefringence: float, wavelength: float) -> float:
    """Phase retardance (rad) between the fast and slow axes of a wave plate:
    2 pi (delta_n) thickness / lambda. pi = half-wave (flips polarization angle),
    pi/2 = quarter-wave (linear -> circular)."""
    return 2.0 * math.pi * birefringence * thickness / wavelength
