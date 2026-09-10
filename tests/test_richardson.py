"""Tests for richardson: stratified-shear turbulence and Kelvin-Helmholtz onset."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import richardson as ri

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Gradient Richardson Ri = N^2 / shear^2.
check("Ri = N^2 / shear^2", abs(ri.gradient_richardson(1e-4, 0.02) - 1e-4 / 0.0004) < 1e-9)

# Strong stratification, weak shear -> stable (Ri > 1/4).
Ri_stable = ri.gradient_richardson(1e-4, 0.01)   # N^2=1e-4, shear=0.01 -> Ri=1.0
check("weak shear is KH-stable", ri.is_kh_stable(Ri_stable))
check("Ri=1 stable", Ri_stable == 1.0)

# Strong shear -> Ri below 1/4 -> unstable.
Ri_unstable = ri.gradient_richardson(1e-4, 0.05)  # -> Ri = 0.04
check("strong shear below critical", Ri_unstable < 0.25)
check("strong shear KH-unstable", not ri.is_kh_stable(Ri_unstable))

# Threshold at exactly 1/4.
check("Ri=0.25 marginal (not >0.25)", not ri.is_kh_stable(0.25))
check("Ri just above 0.25 stable", ri.is_kh_stable(0.2500001))

# Bulk Richardson: g (drho/rho) L / du^2.
Rib = ri.bulk_richardson(1.0, 1000.0, 10.0, 0.5)
check("bulk Ri = g drho/rho L / du^2",
      abs(Rib - 9.80665 * (1.0 / 1000.0) * 10.0 / 0.25) < 1e-6)
# Bigger velocity jump -> smaller bulk Ri (more likely to mix).
check("bulk Ri falls with velocity jump",
      ri.bulk_richardson(1.0, 1000.0, 10.0, 1.0) < Rib)
# Bigger density jump -> larger bulk Ri (more stable).
check("bulk Ri rises with density jump",
      ri.bulk_richardson(2.0, 1000.0, 10.0, 0.5) > Rib)

# Brunt-Vaisala frequency: sqrt(N^2), zero if unstable.
check("N = sqrt(N^2)", abs(ri.brunt_vaisala_frequency(1e-4) - 0.01) < 1e-9)
check("N = 0 for unstable layer", ri.brunt_vaisala_frequency(-1e-4) == 0.0)

# Critical shear: du/dz that brings Ri to 1/4.
N2 = 1e-4
sc = ri.critical_shear(N2)
check("critical shear gives Ri = 1/4", abs(ri.gradient_richardson(N2, sc) - 0.25) < 1e-9)
# Shear steeper than critical destabilizes.
check("steeper than critical is unstable", not ri.is_kh_stable(ri.gradient_richardson(N2, 2 * sc)))
check("gentler than critical is stable", ri.is_kh_stable(ri.gradient_richardson(N2, 0.5 * sc)))

# N^2 from density gradient: stable when density decreases upward.
check("N^2 > 0 for density decreasing upward",
      ri.n_squared_from_density(-0.001, 1000.0) > 0.0)
check("N^2 < 0 for density increasing upward (unstable)",
      ri.n_squared_from_density(0.001, 1000.0) < 0.0)

# A realistic ocean thermocline: N ~ 0.01 rad/s, shear 0.005 /s -> Ri = 4 (stable, layered).
N2_thermo = 0.01 ** 2
check("stable thermocline Ri ~4",
      abs(ri.gradient_richardson(N2_thermo, 0.005) - 4.0) < 0.01)
check("stable thermocline is layered", ri.is_kh_stable(ri.gradient_richardson(N2_thermo, 0.005)))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all richardson tests passed")
