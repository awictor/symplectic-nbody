"""Tests for blasius: the laminar flat-plate boundary layer."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import blasius as bl

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Air over a plate: U = 10 m/s, nu = 1.5e-5 m^2/s.
U, NU, RHO = 10.0, 1.5e-5, 1.225

# Local Reynolds number at x = 0.1 m: 10*0.1/1.5e-5 ~ 66667.
check("Re_x = U x / nu", abs(bl.reynolds_x(U, 0.1, NU) - 10.0 * 0.1 / 1.5e-5) < 1e-3)

# Boundary layer at x=0.1 m: delta = 5*0.1/sqrt(66667) ~ 1.94 mm.
d = bl.bl_thickness(U, 0.1, NU)
check("BL thickness ~1.9 mm at 10 cm", 0.0018 < d < 0.0021)

# Thickness grows as sqrt(x): quadruple x -> double delta.
check("thickness doubles when x quadruples",
      abs(bl.bl_thickness(U, 0.4, NU) - 2.0 * bl.bl_thickness(U, 0.1, NU)) < 1e-9)

# Thickness ordering: delta > delta* > theta (5.0 > 1.721 > 0.664).
x = 0.2
check("delta > delta* > theta",
      bl.bl_thickness(U, x, NU) > bl.displacement_thickness(U, x, NU)
      > bl.momentum_thickness(U, x, NU))
# Ratios are fixed constants.
check("delta*/delta = 1.721/5.0",
      abs(bl.displacement_thickness(U, x, NU) / bl.bl_thickness(U, x, NU) - 1.721 / 5.0) < 1e-9)
check("theta/delta = 0.664/5.0",
      abs(bl.momentum_thickness(U, x, NU) / bl.bl_thickness(U, x, NU) - 0.664 / 5.0) < 1e-9)

# Local skin friction c_f = 0.664/sqrt(Re_x); at x=0.1 ~ 0.00257.
cf = bl.skin_friction_local(U, 0.1, NU)
check("local c_f ~0.00257 at 10 cm", 0.0024 < cf < 0.0028)
check("c_f falls with distance", bl.skin_friction_local(U, 0.4, NU) < bl.skin_friction_local(U, 0.1, NU))

# Average C_D = twice the local c_f at the trailing edge.
L = 0.5
check("average C_D = 2 * local c_f at L",
      abs(bl.skin_friction_average(U, L, NU) - 2.0 * bl.skin_friction_local(U, L, NU)) < 1e-9)
check("average C_D = 1.328/sqrt(Re_L)",
      abs(bl.skin_friction_average(U, L, NU) - 1.328 / math.sqrt(bl.reynolds_x(U, L, NU))) < 1e-12)

# Drag force positive and scales with width.
F1 = bl.drag_force(U, L, 0.1, NU, RHO)
F2 = bl.drag_force(U, L, 0.2, NU, RHO)
check("drag positive", F1 > 0.0)
check("drag doubles with width", abs(F2 - 2.0 * F1) < 1e-9)
# Drag scales as U^1.5 (C_D ~ U^-0.5 times U^2).
check("drag scales as U^1.5",
      abs(bl.drag_force(4 * U, L, 0.1, NU, RHO) / bl.drag_force(U, L, 0.1, NU, RHO)
          - 4.0 ** 1.5) < 1e-6)

# Transition: Re_x hits 5e5 at x = 5e5 * nu / U = 0.75 m for these values.
xt = bl.transition_distance(U, NU)
check("transition at x = 0.75 m", abs(xt - 0.75) < 1e-6)
check("Re_x = 5e5 at transition", abs(bl.reynolds_x(U, xt, NU) - 5e5) < 1.0)
# Faster flow -> earlier transition.
check("faster flow transitions sooner", bl.transition_distance(2 * U, NU) < xt)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all blasius tests passed")
