"""The Franck-Hertz experiment: seeing atomic energy levels one collision at a time.

In 1914 Franck and Hertz fired electrons through mercury vapour and watched the collected
current as they ramped up the accelerating voltage. The current did not rise smoothly -- it
climbed, then dropped sharply, then climbed again, in a regular sawtooth. The drops came at
evenly spaced voltages, 4.9 V apart for mercury. The explanation clinched the Bohr atom:
electrons collide *elastically* with atoms (losing no energy) until they gain exactly the
excitation energy of the atom, at which point they can collide *inelastically*, dump that
quantum, and are left too slow to reach the collector. The spacing between current dips is
therefore the atom's first excitation energy in volts,

    e * dV = E_excitation,        dV = E_ex / e,

direct proof that atomic energy is quantized. A fast electron can excite the atom several
times on its way across, so the dips repeat at V = V_0 + n dV.

The atom then relaxes by emitting a photon at that energy, lambda = h c / E_ex -- 254 nm for
mercury's 4.9 eV, the ultraviolet line -- tying the collision experiment to the emission
spectrum. This module gives the excitation voltage from an energy, the voltages of the
successive current dips, the number of excitations an electron of a given energy can drive,
and the emission wavelength, and reproduces mercury's 4.9 V spacing and 254 nm line. SI units
with eV helpers. Pure stdlib; the atomic-quantization companion to the Bohr and
photoelectric notes.
"""

from __future__ import annotations

E_CHARGE = 1.602176634e-19    # C (also J per eV)
H = 6.62607015e-34           # J s
C = 299792458.0              # m/s

HG_EXCITATION_EV = 4.9        # mercury first excitation energy (eV)


def excitation_voltage(energy_ev: float) -> float:
    """Accelerating voltage dV = E_ex / e at which electrons first excite the atom -- equal
    numerically to the excitation energy in eV (4.9 V for mercury)."""
    return energy_ev            # eV / e in volts is the same number


def dip_voltages(excitation_ev: float, n_dips: int, offset: float = 0.0) -> list:
    """Voltages of the successive Franck-Hertz current dips: V_n = offset + n * dV, for
    n = 1..n_dips, spaced by the excitation voltage. The offset accounts for the contact
    potential."""
    dV = excitation_voltage(excitation_ev)
    return [offset + n * dV for n in range(1, n_dips + 1)]


def dip_spacing(excitation_ev: float) -> float:
    """Spacing between adjacent current dips (V) -- the atom's excitation energy in volts,
    the quantity the experiment measures. 4.9 V for mercury."""
    return excitation_voltage(excitation_ev)


def num_excitations(accel_voltage: float, excitation_ev: float) -> int:
    """How many times an electron accelerated through accel_voltage can inelastically excite
    the atom before it runs out of energy: floor(V / dV)."""
    return int(accel_voltage / excitation_voltage(excitation_ev))


def emission_wavelength(excitation_ev: float) -> float:
    """Wavelength lambda = h c / E (m) of the photon the excited atom emits on relaxing.
    254 nm (UV) for mercury's 4.9 eV."""
    energy_j = excitation_ev * E_CHARGE
    return H * C / energy_j


def residual_energy(accel_voltage: float, excitation_ev: float) -> float:
    """Kinetic energy (eV) an electron retains after exciting the atom as many times as it
    can: V - n dV, the leftover that determines whether it reaches the collector."""
    n = num_excitations(accel_voltage, excitation_ev)
    return accel_voltage - n * excitation_voltage(excitation_ev)
