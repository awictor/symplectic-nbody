"""Tests for hall_effect: the Hall voltage and carrier characterization."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hall_effect as he

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Copper: n ~ 8.5e28 /m^3. A 1 mm-thick bar, 1 A, 1 T -> V_H ~ 0.07 uV (tiny, as observed).
N_CU = 8.5e28
V = he.hall_voltage(1.0, 1.0, N_CU, 1e-3)
check("copper Hall voltage ~0.07 uV", 5e-8 < V < 1e-7)

# V_H = I B / (n q t): scales with I and B, inverse with n and t.
check("V_H linear in current", abs(he.hall_voltage(2.0, 1.0, N_CU, 1e-3) - 2.0 * V) < 1e-16)
check("V_H linear in field", abs(he.hall_voltage(1.0, 2.0, N_CU, 1e-3) - 2.0 * V) < 1e-16)
check("V_H inverse in thickness", abs(he.hall_voltage(1.0, 1.0, N_CU, 2e-3) - V / 2.0) < 1e-16)
check("thinner sample, larger V_H", he.hall_voltage(1.0, 1.0, N_CU, 1e-4) > V)

# carrier_density inverts hall_voltage.
n_back = he.carrier_density(1.0, 1.0, V, 1e-3)
check("carrier_density inverts hall_voltage", abs(n_back - N_CU) / N_CU < 1e-9)

# Hall coefficient sign: negative for electrons (q = -e), positive for holes (q = +e).
R_elec = he.hall_coefficient(N_CU, charge=-he.E_CHARGE)
R_hole = he.hall_coefficient(N_CU, charge=he.E_CHARGE)
check("electron Hall coeff negative", R_elec < 0.0)
check("hole Hall coeff positive", R_hole > 0.0)
check("carrier_sign reads electrons", he.carrier_sign(R_elec) == "electrons")
check("carrier_sign reads holes", he.carrier_sign(R_hole) == "holes")

# R_H magnitude = 1/(n e); copper ~ 7.4e-11 m^3/C.
check("copper |R_H| ~7e-11", 6e-11 < abs(R_elec) < 9e-11)
# Lower carrier density (semiconductor) -> much larger |R_H|.
check("semiconductor larger |R_H|",
      abs(he.hall_coefficient(1e22)) > abs(he.hall_coefficient(N_CU)))

# Mobility mu = |R_H| sigma. Copper sigma ~ 6e7 S/m -> mu ~ 4e-3 m^2/Vs.
mu_cu = he.hall_mobility(abs(R_elec), 6e7)
check("copper mobility ~4e-3 m^2/Vs", 3e-3 < mu_cu < 6e-3)

# Hall angle: arctan(mu B); small for a metal at 1 T, larger for high-mobility semiconductor.
check("copper Hall angle small at 1 T", he.hall_angle(mu_cu, 1.0) < math.radians(1.0))
# High-mobility 2DEG (mu ~ 10 m^2/Vs) at 1 T -> large Hall angle.
check("high-mobility Hall angle large", he.hall_angle(10.0, 1.0) > math.radians(80.0))
check("Hall angle rises with field",
      he.hall_angle(0.1, 5.0) > he.hall_angle(0.1, 1.0))
# Hall angle -> 0 as mu B -> 0.
check("Hall angle ~0 for tiny mu B", he.hall_angle(1e-6, 1.0) < 1e-5)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all hall_effect tests passed")
