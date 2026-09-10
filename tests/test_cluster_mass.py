"""Tests for cluster_mass: the virial mass estimator, checked against Coma."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cluster_mass as cm

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Coma cluster: sigma_los ~ 1000 km/s, R ~ 1.5 Mpc -> M ~ 1e15 solar masses.
M_coma = cm.virial_mass_los(1000.0, 1.5)
check("Coma dynamical mass ~1e15 Msun", 5e14 < M_coma / cm.M_SUN < 1e16)

# virial_mass scales as sigma^2 and linearly with R.
check("mass scales as sigma^2",
      abs(cm.virial_mass(2000e3, 1e22) - 4.0 * cm.virial_mass(1000e3, 1e22)) < 1e30)
check("mass scales with radius",
      abs(cm.virial_mass(1000e3, 2e22) - 2.0 * cm.virial_mass(1000e3, 1e22)) < 1e30)

# LOS estimator uses sigma^2 = 3 sigma_los^2, so it is 3x a naive sigma=sigma_los mass.
naive = cm.virial_mass(1000e3, 1.5 * cm.MPC)      # treats sigma_los as the full 3-D sigma
check("LOS mass = 3x the naive sigma_los mass",
      abs(cm.virial_mass_los(1000.0, 1.5) - 3.0 * naive) < 1e-3 * cm.virial_mass_los(1000.0, 1.5))

# alpha scaling.
check("mass linear in alpha",
      abs(cm.virial_mass(1e6, 1e22, alpha=10.0) - 2.0 * cm.virial_mass(1e6, 1e22, alpha=5.0)) < 1e30)

# Escape velocity from a 1e15 Msun cluster within 1.5 Mpc is a few thousand km/s.
v_esc = cm.escape_velocity(1e15 * cm.M_SUN, 1.5 * cm.MPC)
check("cluster escape velocity 1000-4000 km/s", 1e6 < v_esc < 4e6)
check("escape velocity sqrt scaling",
      abs(cm.escape_velocity(4e15 * cm.M_SUN, 1.5 * cm.MPC)
          - 2.0 * cm.escape_velocity(1e15 * cm.M_SUN, 1.5 * cm.MPC)) < 1e3)

# Crossing time of Coma ~ R/sigma ~ 1.5 Mpc / (1732 km/s) ~ 2-3 Gyr (< Hubble time).
sigma3d = math.sqrt(3.0) * 1000e3
tc = cm.crossing_time(1.5 * cm.MPC, sigma3d)
tc_gyr = tc / (1e9 * 365.25 * 86400)
check("Coma crossing time ~1 Gyr", 0.5 < tc_gyr < 2.0)
check("crossing time much less than Hubble time (relaxed)", tc_gyr < 13.8)

# Mass-to-light: Coma M ~ 1e15 Msun, L ~ 5e12 Lsun -> M/L ~ 200 (dark-matter regime).
ml = cm.mass_to_light(1e15 * cm.M_SUN, 5e12)
check("Coma M/L ~ 200 (dark matter)", 100.0 < ml < 400.0)
check("stellar M/L a few for stars alone",
      cm.mass_to_light(5e12 * cm.M_SUN, 5e12) < 10.0)

# Dark-matter fraction: if only ~5% of the mass is luminous, fraction ~0.95.
f_dm = cm.dark_matter_fraction(1e15 * cm.M_SUN, 5e13 * cm.M_SUN)
check("dark-matter fraction ~0.95", 0.9 < f_dm < 0.97)
check("no dark matter if luminous = dynamical",
      abs(cm.dark_matter_fraction(1e15 * cm.M_SUN, 1e15 * cm.M_SUN)) < 1e-12)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all cluster_mass tests passed")
