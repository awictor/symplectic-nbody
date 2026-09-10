"""Tests for convection: Newton cooling, Nusselt correlations, lumped cooling."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import convection as cv

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Newton cooling: flux is h*dT, positive when surface hotter.
check("Newton flux = h dT", abs(cv.newton_cooling_flux(10.0, 350.0, 300.0) - 500.0) < 1e-9)
check("no flux at thermal equilibrium", cv.newton_cooling_flux(10.0, 300.0, 300.0) == 0.0)
check("flux reverses when fluid hotter", cv.newton_cooling_flux(10.0, 300.0, 350.0) < 0.0)

# h from Nusselt: h = Nu k / L.
check("h = Nu k / L", abs(cv.heat_transfer_coefficient(100.0, 0.6, 0.05) - 100.0 * 0.6 / 0.05) < 1e-9)

# Dittus-Boelter for water in a pipe: Re=1e4, Pr=7 -> Nu ~ 76, h with k=0.6, D=0.02 -> ~2000+.
nu_water = cv.dittus_boelter(1e4, 7.0, heating=True)
check("Dittus-Boelter water Nu ~76", 70.0 < nu_water < 82.0)
h_water = cv.heat_transfer_coefficient(nu_water, 0.6, 0.02)
check("forced water h in the thousands", 2000.0 < h_water < 2600.0)

# Heating exponent (0.4) gives higher Nu than cooling (0.3) at Pr>1.
check("heating Nu > cooling Nu for Pr>1",
      cv.dittus_boelter(1e4, 7.0, True) > cv.dittus_boelter(1e4, 7.0, False))

# Dittus-Boelter scales as Re^0.8.
check("Nu scales as Re^0.8",
      abs(cv.dittus_boelter(2e4, 7.0) / cv.dittus_boelter(1e4, 7.0) - 2.0 ** 0.8) < 1e-6)

# Flat-plate laminar: Nu = 0.664 Re^0.5 Pr^(1/3); air Re=1e5, Pr=0.7 -> ~187.
nu_plate = cv.flat_plate_laminar(1e5, 0.7)
check("flat-plate air Nu ~187", 180.0 < nu_plate < 195.0)
check("flat-plate scales as sqrt(Re)",
      abs(cv.flat_plate_laminar(4e5, 0.7) / cv.flat_plate_laminar(1e5, 0.7) - 2.0) < 1e-6)

# Biot number: small for a conductive metal in gentle air (lumped OK), large otherwise.
Bi_metal = cv.biot_number(10.0, 0.01, 200.0)      # aluminium in air
check("metal in air is lumped (Bi < 0.1)", Bi_metal < 0.1)
Bi_bad = cv.biot_number(1000.0, 0.1, 1.0)         # forced water, poor conductor, thick
check("forced water on insulator not lumped (Bi > 1)", Bi_bad > 1.0)

# Lumped cooling: time constant tau = rho c_p V / (h A) and exponential decay.
tau = cv.lumped_time_constant(2700.0, 900.0, 1e-6, 10.0, 6e-4)
# T decays from 100 C toward 20 C; after one tau, excess is 1/e of initial.
T0, Tinf = 100.0, 20.0
T_at_tau = cv.lumped_temperature(tau, T0, Tinf, tau)
check("lumped T after one tau is 1/e of excess",
      abs((T_at_tau - Tinf) / (T0 - Tinf) - math.exp(-1.0)) < 1e-9)
check("lumped starts at T0", abs(cv.lumped_temperature(0.0, T0, Tinf, tau) - T0) < 1e-9)
check("lumped approaches fluid temperature",
      abs(cv.lumped_temperature(20.0 * tau, T0, Tinf, tau) - Tinf) < 0.1)
check("cooling is monotonic",
      cv.lumped_temperature(tau, T0, Tinf, tau) > cv.lumped_temperature(2 * tau, T0, Tinf, tau))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all convection tests passed")
