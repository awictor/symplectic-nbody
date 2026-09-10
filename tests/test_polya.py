"""Tests for polya: random-walk recurrence and return probability."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import polya

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Return probability: 1 in 1D/2D, ~0.34 in 3D, falling higher.
check("1D returns with probability 1", polya.return_probability(1) == 1.0)
check("2D returns with probability 1", polya.return_probability(2) == 1.0)
check("3D return probability ~0.34", abs(polya.return_probability(3) - 0.3405) < 0.001)
check("4D return probability ~0.19", abs(polya.return_probability(4) - 0.1932) < 0.001)
check("return probability falls with dimension",
      polya.return_probability(4) < polya.return_probability(3))

# Recurrence classification (Polya's theorem).
check("1D recurrent", polya.is_recurrent(1))
check("2D recurrent", polya.is_recurrent(2))
check("3D transient", not polya.is_recurrent(3))
check("5D transient", not polya.is_recurrent(5))

# Expected visits: infinite when recurrent, finite when transient.
check("infinite visits in 1D", math.isinf(polya.expected_visits(1)))
check("infinite visits in 2D", math.isinf(polya.expected_visits(2)))
check("finite visits in 3D", not math.isinf(polya.expected_visits(3)))
check("3D expected visits ~1.516", abs(polya.expected_visits(3) - 1.516) < 0.01)
check("fewer visits in higher dimension", polya.expected_visits(4) < polya.expected_visits(3))

# Escape probability: 0 for d<=2, ~0.66 in 3D.
check("no escape in 2D", polya.escape_probability(2) == 0.0)
check("3D escape ~0.66", abs(polya.escape_probability(3) - 0.659) < 0.001)
check("escape + return = 1", abs(polya.escape_probability(3) + polya.return_probability(3) - 1.0) < 1e-9)
check("more escape in higher dimension", polya.escape_probability(4) > polya.escape_probability(3))

# Mean-square displacement = n a^2 (diffusive).
check("MSD = n a^2", abs(polya.mean_square_displacement(100, 1.0) - 100.0) < 1e-9)
check("MSD linear in steps",
      abs(polya.mean_square_displacement(200) - 2 * polya.mean_square_displacement(100)) < 1e-9)
check("rms distance ~ sqrt(n)",
      abs(math.sqrt(polya.mean_square_displacement(400)) - 20.0) < 1e-9)

# Simulation: 1D walks almost all return within many steps; 3D fewer.
frac_1d = polya.simulate_return_fraction(1, max_steps=3000, trials=100, seed=1)
frac_3d = polya.simulate_return_fraction(3, max_steps=3000, trials=100, seed=1)
check("1D walks mostly return", frac_1d > 0.85)
check("3D walks return less often than 1D", frac_3d < frac_1d)
# 3D simulated return fraction is in the right ballpark (finite max_steps undercounts).
check("3D simulated return fraction below 0.6", frac_3d < 0.6)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all polya tests passed")
