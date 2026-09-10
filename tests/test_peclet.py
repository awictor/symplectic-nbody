"""Tests for peclet: advection-vs-diffusion dimensionless groups."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import peclet as pe

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Peclet = U L / D.
check("Pe = U L / D", abs(pe.peclet(0.1, 0.01, 1e-9) - 0.1 * 0.01 / 1e-9) < 1e-6)

# A stirred cup (U~0.1 m/s, L~0.05 m, solute D~1e-9) is hugely advective.
check("stirred cup Pe >> 1", pe.peclet(0.1, 0.05, 1e-9) > 1e6)

# Transport inside a cell (U~0, L~1e-5, D~1e-9) with a tiny drift is diffusion-limited.
check("cell-scale slow drift Pe << 1", pe.peclet(1e-7, 1e-5, 1e-9) < 1.0)

# Thermal diffusivity of water: k=0.6, rho=1000, cp=4180 -> ~1.4e-7 m^2/s.
alpha_water = pe.thermal_diffusivity(0.6, 1000.0, 4180.0)
check("water thermal diffusivity ~1.4e-7", 1.3e-7 < alpha_water < 1.5e-7)

# Prandtl of water ~7 (nu ~ 1e-6 m^2/s).
Pr_water = pe.prandtl(1e-6, alpha_water)
check("water Pr ~7", 6.0 < Pr_water < 8.0)

# Prandtl of air ~0.7 (nu ~ 1.5e-5, alpha ~ 2.1e-5).
alpha_air = pe.thermal_diffusivity(0.026, 1.2, 1005.0)
Pr_air = pe.prandtl(1.5e-5, alpha_air)
check("air Pr ~0.7", 0.6 < Pr_air < 0.8)

# Schmidt of a small molecule in water ~1000 (nu 1e-6, D 1e-9).
Sc = pe.schmidt(1e-6, 1e-9)
check("aqueous solute Sc ~1000", 900.0 < Sc < 1100.0)

# Lewis number = Sc / Pr = alpha / D.
Le = pe.lewis(alpha_water, 1e-9)
check("Lewis = alpha/D", abs(Le - alpha_water / 1e-9) < 1e-6)
check("Lewis = Sc/Pr", abs(Le - Sc / Pr_water) < 1e-3 * Le)

# Pe = Re * Pr (heat) and Re * Sc (mass).
check("Pe = Re*Pr", abs(pe.peclet_from_reynolds(2000.0, Pr_water) - 2000.0 * Pr_water) < 1e-6)

# Cross-check: Pe_mass computed directly equals Re*Sc.
U, L, nu, D = 0.01, 0.001, 1e-6, 1e-9
Re = U * L / nu
check("direct Pe equals Re*Sc",
      abs(pe.peclet(U, L, D) - pe.peclet_from_reynolds(Re, pe.schmidt(nu, D))) < 1e-3)

# Crossover length L = D/U, and Pe(L_cross) = 1.
Lc = pe.crossover_length(0.01, 1e-9)
check("crossover length = D/U", abs(Lc - 1e-9 / 0.01) < 1e-18)
check("Pe = 1 at crossover length", abs(pe.peclet(0.01, Lc, 1e-9) - 1.0) < 1e-9)

# Faster flow -> larger Peclet; larger diffusivity -> smaller Peclet.
check("Pe rises with velocity", pe.peclet(0.2, 0.01, 1e-9) > pe.peclet(0.1, 0.01, 1e-9))
check("Pe falls with diffusivity", pe.peclet(0.1, 0.01, 1e-8) < pe.peclet(0.1, 0.01, 1e-9))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all peclet tests passed")
