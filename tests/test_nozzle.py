"""Tests for nozzle: isentropic de Laval flow, choking, area-Mach relation."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import nozzle as nz

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Isentropic ratios all equal 1 at M=0 (static = stagnation at rest).
check("ratios = 1 at M=0", abs(nz.temperature_ratio(0.0) - 1.0) < 1e-12
      and abs(nz.pressure_ratio(0.0) - 1.0) < 1e-12
      and abs(nz.density_ratio(0.0) - 1.0) < 1e-12)

# At M=1 (air): T0/T = 1.2, P0/P ~ 1.893, rho0/rho ~ 1.577.
check("T0/T = 1.2 at M=1", abs(nz.temperature_ratio(1.0) - 1.2) < 1e-9)
check("P0/P ~1.893 at M=1", abs(nz.pressure_ratio(1.0) - 1.8929) < 1e-3)

# Critical pressure ratio for air ~0.5283.
check("critical pressure ratio ~0.528", abs(nz.critical_pressure_ratio() - 0.5283) < 1e-3)
check("critical ratio = 1/(P0/P at M=1)",
      abs(nz.critical_pressure_ratio() - 1.0 / nz.pressure_ratio(1.0)) < 1e-9)

# Area ratio is minimum (=1) at the throat, M=1.
check("A/A* = 1 at M=1", abs(nz.area_ratio(1.0) - 1.0) < 1e-9)
# Both subsonic and supersonic sides need more area than the throat.
check("A/A* > 1 subsonic", nz.area_ratio(0.5) > 1.0)
check("A/A* > 1 supersonic", nz.area_ratio(2.0) > 1.0)
# Known value: A/A* at M=2 (air) ~ 1.6875.
check("A/A* ~1.6875 at M=2", abs(nz.area_ratio(2.0) - 1.6875) < 1e-3)
# A/A* at M=3 ~ 4.2346.
check("A/A* ~4.235 at M=3", abs(nz.area_ratio(3.0) - 4.2346) < 1e-3)

# Invert area ratio: supersonic branch recovers M=2.5.
ar = nz.area_ratio(2.5)
check("supersonic inversion recovers M=2.5", abs(nz.mach_from_area_ratio(ar, True) - 2.5) < 1e-4)
# Subsonic branch recovers M=0.4 from the same-shaped relation.
ar_sub = nz.area_ratio(0.4)
check("subsonic inversion recovers M=0.4", abs(nz.mach_from_area_ratio(ar_sub, False) - 0.4) < 1e-4)
# Same area ratio gives two different Mach numbers (sub vs super).
check("one area ratio, two Mach branches",
      nz.mach_from_area_ratio(2.0, True) > 1.0 > nz.mach_from_area_ratio(2.0, False))

# Bigger area ratio -> higher exit Mach number.
check("larger area ratio, higher exit Mach",
      nz.mach_from_area_ratio(10.0, True) > nz.mach_from_area_ratio(2.0, True))

# Choked mass flow scales linearly with chamber pressure and throat area.
mdot1 = nz.choked_mass_flow(1e6, 3000.0, 0.01)
mdot2 = nz.choked_mass_flow(2e6, 3000.0, 0.01)
check("choked flow doubles with chamber pressure", abs(mdot2 - 2.0 * mdot1) < 1e-6)
mdot3 = nz.choked_mass_flow(1e6, 3000.0, 0.02)
check("choked flow doubles with throat area", abs(mdot3 - 2.0 * mdot1) < 1e-6)
check("choked flow is positive", mdot1 > 0.0)

# Exhaust velocity rises with exit Mach; a hot chamber expanded to M=3 gives km/s speeds.
v_exit = nz.exhaust_velocity(3000.0, 3.0)
check("hot chamber M=3 exhaust > 1.5 km/s", v_exit > 1500.0)
check("higher exit Mach, faster exhaust",
      nz.exhaust_velocity(3000.0, 4.0) > nz.exhaust_velocity(3000.0, 2.0))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all nozzle tests passed")
