"""Tests for strouhal: von Karman vortex-shedding frequency and lock-in."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import strouhal as st

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Shedding frequency f = St U / d. Wind 10 m/s over a 5 mm wire, St=0.2 -> 400 Hz.
f = st.shedding_frequency(10.0, 0.005)
check("5 mm wire in 10 m/s wind sheds ~400 Hz", 395.0 < f < 405.0)

# f = St U / d exactly.
check("f = St U / d", abs(st.shedding_frequency(20.0, 0.01) - 0.2 * 20.0 / 0.01) < 1e-9)

# Frequency scales linearly with speed, inversely with size.
check("frequency doubles with speed",
      abs(st.shedding_frequency(20.0, 0.01) - 2.0 * st.shedding_frequency(10.0, 0.01)) < 1e-9)
check("frequency halves with size",
      abs(st.shedding_frequency(10.0, 0.02) - 0.5 * st.shedding_frequency(10.0, 0.01)) < 1e-9)

# strouhal_number inverts shedding_frequency.
check("strouhal_number recovers St",
      abs(st.strouhal_number(f, 0.005, 10.0) - 0.2) < 1e-9)

# Roshko: St rises toward ~0.21 with Re; ~0.2 at Re=2000, higher at Re=1e5.
check("Roshko St ~0.2 at Re=2000", 0.19 < st.roshko_strouhal(2000.0) < 0.212)
check("Roshko St rises with Re", st.roshko_strouhal(1e5) > st.roshko_strouhal(2000.0))
check("Roshko St approaches 0.212", st.roshko_strouhal(1e6) > 0.211)
# At the low-Re cutoff Re=21.2 the formula gives zero (shedding onset).
check("Roshko St = 0 at Re=21.2", abs(st.roshko_strouhal(21.2)) < 1e-9)

# Aeolian tone equals the shedding frequency.
check("aeolian tone = shedding frequency",
      st.aeolian_frequency(10.0, 0.005) == st.shedding_frequency(10.0, 0.005))

# Lock-in: the speed whose shedding frequency matches a natural frequency.
# A chimney d=2 m, f_n=0.5 Hz -> U = 0.5*2/0.2 = 5 m/s.
U_lock = st.lock_in_velocity(0.5, 2.0)
check("chimney lock-in at ~5 m/s", abs(U_lock - 5.0) < 1e-6)
# Round trip: shedding at the lock-in speed equals the natural frequency.
check("shedding at lock-in = natural frequency",
      abs(st.shedding_frequency(U_lock, 2.0) - 0.5) < 1e-9)

# Vortex spacing = d / St ~ 5 diameters for St=0.2.
check("vortex spacing ~5 diameters",
      abs(st.vortex_spacing(10.0, 0.01) - 0.01 / 0.2) < 1e-12)
check("spacing independent of speed",
      abs(st.vortex_spacing(50.0, 0.01) - st.vortex_spacing(10.0, 0.01)) < 1e-12)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all strouhal tests passed")
