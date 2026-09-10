"""Tests for rabi: two-level Rabi oscillations and pulses."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import rabi as rb

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


OMEGA = 2 * math.pi * 1e6   # 1 MHz Rabi frequency

# On resonance: full flopping 0 -> 1 -> 0.
check("starts in ground state", abs(rb.excited_probability(OMEGA, 0.0, 0.0)) < 1e-12)
# At pi pulse (Omega t = pi): fully excited.
t_pi = rb.pi_pulse_time(OMEGA)
check("pi pulse fully inverts", abs(rb.excited_probability(OMEGA, 0.0, t_pi) - 1.0) < 1e-9)
# At 2pi: back to ground.
check("2pi returns to ground", abs(rb.excited_probability(OMEGA, 0.0, 2 * t_pi)) < 1e-9)
# At pi/2 pulse: equal superposition (P = 0.5).
t_half = rb.half_pi_pulse_time(OMEGA)
check("pi/2 pulse gives P=0.5", abs(rb.excited_probability(OMEGA, 0.0, t_half) - 0.5) < 1e-9)

# Rabi frequency = d E / hbar.
check("Omega = d E / hbar", abs(rb.rabi_frequency(1e-29, 1e5) - 1e-29 * 1e5 / rb.HBAR) < 1)
check("stronger field, faster flopping",
      rb.rabi_frequency(1e-29, 2e5) > rb.rabi_frequency(1e-29, 1e5))

# Generalized Rabi: sqrt(Omega^2 + delta^2), faster than bare Omega off resonance.
delta = OMEGA   # detune by one Rabi frequency
check("generalized Rabi = sqrt(O^2 + d^2)",
      abs(rb.generalized_rabi(OMEGA, delta) - math.sqrt(2) * OMEGA) < 1)
check("generalized Rabi faster off resonance",
      rb.generalized_rabi(OMEGA, delta) > OMEGA)
check("generalized Rabi = Omega on resonance",
      abs(rb.generalized_rabi(OMEGA, 0.0) - OMEGA) < 1e-6)

# Peak probability: 1 on resonance, Lorentzian off.
check("peak = 1 on resonance", abs(rb.peak_probability(OMEGA, 0.0) - 1.0) < 1e-12)
check("peak = 1/2 at detuning = Omega", abs(rb.peak_probability(OMEGA, OMEGA) - 0.5) < 1e-12)
check("peak falls with detuning",
      rb.peak_probability(OMEGA, 2 * OMEGA) < rb.peak_probability(OMEGA, OMEGA))
# Off resonance the atom never fully inverts.
check("off resonance never fully excited",
      max(rb.excited_probability(OMEGA, delta, t_pi * k / 20) for k in range(41)) < 0.99)

# Off-resonant excitation never exceeds the peak probability.
pk = rb.peak_probability(OMEGA, delta)
worst = max(rb.excited_probability(OMEGA, delta, 3 * t_pi * k / 200) for k in range(201))
check("excitation bounded by peak probability", worst <= pk + 1e-9)

# Stronger drive -> shorter pi pulse.
check("stronger drive, shorter pi pulse",
      rb.pi_pulse_time(2 * OMEGA) < rb.pi_pulse_time(OMEGA))
check("pi/2 pulse is half the pi pulse",
      abs(rb.half_pi_pulse_time(OMEGA) - rb.pi_pulse_time(OMEGA) / 2.0) < 1e-15)

# 1 MHz Rabi frequency -> pi pulse of ~0.5 microseconds.
check("1 MHz Rabi -> ~0.5 us pi pulse", 4e-7 < rb.pi_pulse_time(OMEGA) < 6e-7)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all rabi tests passed")
