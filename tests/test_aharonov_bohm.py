"""Tests for aharonov_bohm: flux-driven quantum phase with no field on the path."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import aharonov_bohm as ab

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Flux quantum h/e ~ 4.14e-15 Wb; superconducting h/2e ~ 2.07e-15.
check("flux quantum ~4.14e-15 Wb", abs(ab.flux_quantum() - 4.136e-15) < 0.01e-15)
check("SC flux quantum ~2.07e-15 Wb", abs(ab.FLUX_QUANTUM_SC - 2.068e-15) < 0.01e-15)
check("SC flux quantum is half the electron one",
      abs(ab.FLUX_QUANTUM_SC - ab.FLUX_QUANTUM / 2.0) < 1e-20)

# One flux quantum advances the phase by exactly 2 pi.
check("one flux quantum -> 2 pi phase",
      abs(ab.phase_shift(ab.flux_quantum()) - 2.0 * math.pi) < 1e-6)
# Phase linear in flux.
check("phase linear in flux",
      abs(ab.phase_shift(2e-15) - 2.0 * ab.phase_shift(1e-15)) < 1e-6)
check("zero flux, zero phase", ab.phase_shift(0.0) == 0.0)

# Number of flux quanta = Phi/Phi_0.
check("num flux quanta = Phi/Phi_0",
      abs(ab.num_flux_quanta(3 * ab.flux_quantum()) - 3.0) < 1e-9)
# Larger charge (Cooper pair) sees twice the quanta for the same flux.
check("Cooper pair sees twice the quanta",
      abs(ab.num_flux_quanta(1e-15, 2 * ab.E_CHARGE) - 2 * ab.num_flux_quanta(1e-15)) < 1e-6)

# Fringe shift is periodic: integer quanta -> back to zero.
check("integer quanta -> fringe shift 0",
      abs(ab.fringe_shift(5 * ab.flux_quantum())) < 1e-9)
check("half quantum -> fringe shift 0.5",
      abs(ab.fringe_shift(0.5 * ab.flux_quantum()) - 0.5) < 1e-9)
check("fringe shift in [0,1)", 0.0 <= ab.fringe_shift(3.7 * ab.flux_quantum()) < 1.0)

# Flux through a loop = B A.
check("flux = B A", abs(ab.flux_through_loop(0.01, 1e-4) - 0.01 * 1e-4) < 1e-12)

# Field for one quantum: tiny for a big loop, since Phi_0 is small.
# 1 cm^2 loop: B = 4.14e-15 / 1e-4 = 4.14e-11 T (well below Earth's ~5e-5 T).
B1 = ab.field_for_one_quantum(1e-4)
check("1 cm^2 loop: one quantum at ~4e-11 T", 3e-11 < B1 < 5e-11)
check("bigger loop, smaller field per quantum",
      ab.field_for_one_quantum(1e-2) < ab.field_for_one_quantum(1e-4))
# Round-trip: that field through that loop is exactly one flux quantum.
check("field_for_one_quantum gives one quantum",
      abs(ab.num_flux_quanta(ab.flux_through_loop(B1, 1e-4)) - 1.0) < 1e-6)

# A SQUID (Cooper pairs) resolves flux to a fraction of h/2e -- the most sensitive
# magnetometer. Check the SC quantum through a 1 mm^2 loop needs only ~2e-9 T.
check("SC quantum through 1 mm^2 loop ~2e-9 T",
      abs(ab.field_for_one_quantum(1e-6, 2 * ab.E_CHARGE) - 2.07e-9) < 0.1e-9)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all aharonov_bohm tests passed")
