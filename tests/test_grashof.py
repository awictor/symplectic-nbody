"""Tests for grashof: natural convection driven by buoyancy."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import grashof as gr

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Ideal-gas expansion coefficient beta = 1/T; at 300 K ~ 0.00333 /K.
check("beta = 1/T", abs(gr.expansion_coefficient_ideal_gas(300.0) - 1.0 / 300.0) < 1e-12)

# Warm wall in air: dT=20 K, L=0.3 m, beta=1/300, nu=1.5e-5 -> Gr ~ 1.6e8.
beta = gr.expansion_coefficient_ideal_gas(300.0)
Gr = gr.grashof_number(20.0, 0.3, beta, 1.5e-5)
check("warm wall Gr ~1e8", 5e7 < Gr < 5e8)

# Grashof scales as L^3, dT, and 1/nu^2.
check("Gr scales as L^3",
      abs(gr.grashof_number(20.0, 0.6, beta, 1.5e-5) - 8.0 * gr.grashof_number(20.0, 0.3, beta, 1.5e-5)) < 1e-3 * Gr)
check("Gr linear in dT",
      abs(gr.grashof_number(40.0, 0.3, beta, 1.5e-5) - 2.0 * Gr) < 1e-3 * Gr)
check("Gr scales as 1/nu^2",
      abs(gr.grashof_number(20.0, 0.3, beta, 3.0e-5) - Gr / 4.0) < 1e-3 * Gr)

# Rayleigh = Gr Pr; air Pr=0.71.
Ra = gr.rayleigh_number(Gr, 0.71)
check("Ra = Gr Pr", abs(Ra - Gr * 0.71) < 1e-3 * Ra)

# Regime: this wall is laminar (Ra < 1e9).
check("warm wall is laminar (Ra<1e9)", not gr.is_turbulent(Ra))
check("very hot tall wall goes turbulent",
      gr.is_turbulent(gr.rayleigh_number(gr.grashof_number(60.0, 3.0, beta, 1.5e-5), 0.71)))

# Nusselt correlation switches branch at Ra=1e9.
check("laminar branch below 1e9", abs(gr.nusselt_vertical_plate(1e8) - 0.59 * 1e8 ** 0.25) < 1e-6)
check("turbulent branch above 1e9", abs(gr.nusselt_vertical_plate(1e10) - 0.10 * 1e10 ** (1.0/3.0)) < 1e-6)
check("Nu rises with Ra", gr.nusselt_vertical_plate(1e10) > gr.nusselt_vertical_plate(1e6))

# Heat-transfer coefficient for the warm wall: h = Nu k / L ~ a few W/(m^2 K).
Nu = gr.nusselt_vertical_plate(Ra)
h = gr.heat_transfer_coefficient(Nu, 0.026, 0.3)
check("natural-convection h a few W/(m^2 K)", 2.0 < h < 8.0)
check("h far below forced convection (< 20)", h < 20.0)

# Natural vs forced: buoyancy dominates when Gr >> Re^2.
check("still air: buoyancy dominates (Gr/Re^2 >> 1)", gr.natural_vs_forced(1e8, 100.0) > 1.0)
check("strong wind: forced dominates (Gr/Re^2 << 1)", gr.natural_vs_forced(1e8, 1e5) < 1.0)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all grashof tests passed")
