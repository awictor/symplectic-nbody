"""Tests for equipartition: (1/2)kT per DOF and gas heat capacities."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import equipartition as eq

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Energy per degree of freedom is (1/2) k_B T; at 300 K ~ 2.07e-21 J.
check("energy per dof = (1/2)kT", abs(eq.energy_per_dof(300.0) - 0.5 * eq.K_B * 300.0) < 1e-30)
check("energy per dof ~2.07e-21 J at 300 K", 2.0e-21 < eq.energy_per_dof(300.0) < 2.1e-21)

# Monatomic gas: C_V = 3R/2 ~ 12.47, C_P = 5R/2 ~ 20.79, gamma = 5/3.
check("monatomic C_V = 3R/2", abs(eq.molar_cv(3) - 1.5 * eq.R_GAS) < 1e-9)
check("monatomic C_V ~12.47", abs(eq.molar_cv(3) - 12.47) < 0.05)
check("monatomic gamma = 5/3", abs(eq.gamma_ratio(3) - 5.0 / 3.0) < 1e-12)

# Diatomic (room temp, f=5): C_V = 5R/2, gamma = 7/5 = 1.4.
check("diatomic C_V = 5R/2", abs(eq.molar_cv(5) - 2.5 * eq.R_GAS) < 1e-9)
check("diatomic gamma = 7/5", abs(eq.gamma_ratio(5) - 1.4) < 1e-12)

# Mayer's relation C_P - C_V = R for any f.
check("Mayer C_P - C_V = R", abs(eq.molar_cp(5) - eq.molar_cv(5) - eq.R_GAS) < 1e-9)

# Dulong-Petit: solid f=6 -> C_V = 3R ~ 24.94 J/(mol K).
check("Dulong-Petit 3R", abs(eq.molar_cv(6) - 3.0 * eq.R_GAS) < 1e-9)
check("Dulong-Petit ~24.9", abs(eq.molar_cv(6) - 24.94) < 0.05)

# Internal energy scales with dof, moles, T.
check("U = (f/2) nRT", abs(eq.internal_energy(3, 300.0, 2.0)
                           - 1.5 * 2.0 * eq.R_GAS * 300.0) < 1e-6)

# rms speed: N2 (0.028 kg/mol) at 300 K ~ 517 m/s.
check("N2 rms speed ~517 m/s at 300 K", 505.0 < eq.rms_speed(300.0, 0.028) < 525.0)

# Mode activation: full (->1) when T >> theta, frozen (->0) when T << theta.
check("mode fully active when T >> theta", eq.mode_activation(10000.0, 85.0) > 0.99)
check("mode frozen when T << theta", eq.mode_activation(10.0, 6000.0) < 1e-3)
check("mode activation between 0 and 1", 0.0 < eq.mode_activation(300.0, 300.0) < 1.0)

# H2 heat-capacity staircase: theta_rot ~ 85 K, theta_vib ~ 6000 K.
theta_rot, theta_vib = 85.0, 6000.0
cv_cold = eq.effective_cv_diatomic(30.0, theta_rot, theta_vib)     # rotation still freezing
cv_room = eq.effective_cv_diatomic(300.0, theta_rot, theta_vib)    # ~5R/2, vibration frozen
cv_hot = eq.effective_cv_diatomic(5000.0, theta_rot, theta_vib)    # vibration turning on
# At 50 K rotation is only partly on: C_V well below 5R/2.
check("H2 C_V near 3R/2 in the cold", cv_cold < 2.2 * eq.R_GAS)
# At 300 K H2 is the classic 5R/2 (rotation on, vibration off).
check("H2 C_V ~5R/2 at room temperature", abs(cv_room - 2.5 * eq.R_GAS) < 0.05 * eq.R_GAS)
# At 5000 K vibration is significantly active: C_V climbs above 5R/2 toward 7R/2.
check("H2 C_V climbs above 5R/2 when hot", cv_hot > 2.7 * eq.R_GAS)
check("staircase is monotonic in T", cv_cold < cv_room < cv_hot)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all equipartition tests passed")
