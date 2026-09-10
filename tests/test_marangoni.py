"""Tests for marangoni: surface-tension-gradient driven flow."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import marangoni as mg

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Water: dgamma/dT ~ -1.5e-4 N/m/K, mu ~ 1e-3, alpha ~ 1.4e-7.
DG_DT = -1.5e-4
MU, ALPHA = 1e-3, 1.4e-7

# Thin heated water layer: dT=10 K, L=1mm -> Ma large (well above onset).
Ma = mg.marangoni_thermal(DG_DT, 10.0, 1e-3, MU, ALPHA)
check("thin heated layer Ma above onset", Ma > 80.0)
check("Marangoni convects when Ma>80", mg.is_convecting(Ma))

# Uses magnitude of dgamma/dT (sign irrelevant to Ma).
check("Ma uses |dgamma/dT|",
      abs(mg.marangoni_thermal(-DG_DT, 10.0, 1e-3, MU, ALPHA) - Ma) < 1e-6)

# Ma scales linearly with dT, L, and 1/(mu alpha).
check("Ma linear in dT",
      abs(mg.marangoni_thermal(DG_DT, 20.0, 1e-3, MU, ALPHA) - 2.0 * Ma) < 1e-3 * Ma)
check("Ma linear in length",
      abs(mg.marangoni_thermal(DG_DT, 10.0, 2e-3, MU, ALPHA) - 2.0 * Ma) < 1e-3 * Ma)

# Below onset: tiny gradient / thick layer.
Ma_small = mg.marangoni_thermal(DG_DT, 0.01, 1e-4, MU, ALPHA)
check("weak gradient below onset", not mg.is_convecting(Ma_small))

# Solutal Marangoni (tears of wine): dgamma/dc large, mass diffusivity small.
Ma_sol = mg.marangoni_solutal(0.05, 0.1, 1e-3, MU, 1e-9)
check("solutal Marangoni positive and large", Ma_sol > 80.0)

# Surface stress tau = dgamma/dT * dT/dx.
tau = mg.surface_stress(DG_DT, 100.0)     # 100 K/m gradient
check("surface stress = dgamma/dT * dT/dx", abs(tau - DG_DT * 100.0) < 1e-12)
check("stress points toward higher surface tension (cooler side)", tau < 0.0)

# Dynamic Bond number: thick layer on the ground is buoyancy-dominated (Bo_d >> 1).
Bo_thick = mg.dynamic_bond_number(998.0, 9.81, 2e-4, 0.01, DG_DT)
check("thick ground layer buoyancy-dominated (Bo_d>1)", Bo_thick > 1.0)
# Thin layer is Marangoni-dominated (Bo_d << 1).
Bo_thin = mg.dynamic_bond_number(998.0, 9.81, 2e-4, 1e-4, DG_DT)
check("thin layer Marangoni-dominated (Bo_d<1)", Bo_thin < 1.0)
check("Bo_d scales as L^2", abs(mg.dynamic_bond_number(998.0, 9.81, 2e-4, 2e-4, DG_DT)
                                - 4.0 * Bo_thin) < 1e-3 * Bo_thin)
# Microgravity kills buoyancy -> Marangoni always wins.
check("microgravity suppresses buoyancy (Bo_d tiny)",
      mg.dynamic_bond_number(998.0, 1e-5, 2e-4, 0.01, DG_DT) < 1e-3)

# Marangoni velocity: |dgamma/dT| dT / mu, ~ tens of mm/s for a hot water film.
U = mg.marangoni_velocity(DG_DT, 10.0, 1e-3, MU)
check("Marangoni velocity ~1.5 m/s scale", 0.5 < U < 5.0)
check("velocity linear in dT",
      abs(mg.marangoni_velocity(DG_DT, 20.0, 1e-3, MU) - 2.0 * U) < 1e-9)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all marangoni tests passed")
