"""Tests for duffing: the nonlinear double-well oscillator."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import duffing as df

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Potential: single well for alpha>0, double well for alpha<0, beta>0.
check("V(0) = 0", df.potential(0.0, 1.0, 1.0) == 0.0)
check("single well rises from 0", df.potential(1.0, 1.0, 0.0) > 0.0)
# Double well: minima at +/- sqrt(-alpha/beta) = +/-1 for alpha=-1, beta=1.
mins = df.well_minima(-1.0, 1.0)
check("double-well minima at +/-1", abs(mins[0] + 1.0) < 1e-9 and abs(mins[1] - 1.0) < 1e-9)
check("potential lower at min than at 0 (double well)",
      df.potential(1.0, -1.0, 1.0) < df.potential(0.0, -1.0, 1.0))
check("single-well minimum at 0", df.well_minima(1.0, 1.0) == [0.0])

# Regime classification.
check("hardening (alpha>0,beta>0)", df.regime(1.0, 1.0) == "hardening")
check("softening (beta<0)", df.regime(1.0, -1.0) == "softening")
check("double-well (alpha<0,beta>0)", df.regime(-1.0, 1.0) == "double-well")
check("linear (beta=0)", df.regime(1.0, 0.0) == "linear")
check("is_double_well true", df.is_double_well(-1.0, 1.0))
check("is_double_well false for single well", not df.is_double_well(1.0, 1.0))

# Vector field.
dx, dv = df.derivatives((1.0, 0.5), 0.0, 0.1, 1.0, 1.0, 0.0, 1.0)
check("dx = v", abs(dx - 0.5) < 1e-12)
check("dv = -delta v - alpha x - beta x^3", abs(dv - (-0.1 * 0.5 - 1.0 - 1.0)) < 1e-12)

# Undamped linear (beta=0) oscillator conserves amplitude and oscillates at omega~1.
tr = df.trajectory((1.0, 0.0), 0.01, 2000, delta=0.0, alpha=1.0, beta=0.0)
amp = max(abs(p[0]) for p in tr)
check("linear oscillator amplitude ~1", abs(amp - 1.0) < 0.02)

# Damped oscillator decays toward a well.
tr_d = df.trajectory((1.0, 0.0), 0.01, 6000, delta=0.3, alpha=1.0, beta=0.0)
check("damped single-well decays to 0", abs(tr_d[-1][0]) < 0.1)
# Damped double-well settles into one of the wells (+/-1), not the origin.
tr_dw = df.trajectory((0.5, 0.0), 0.01, 8000, delta=0.3, alpha=-1.0, beta=1.0)
check("double-well settles into a well (~+/-1)", abs(abs(tr_dw[-1][0]) - 1.0) < 0.1)

# Backbone: hardening spring's natural frequency rises with amplitude.
f0 = df.backbone_frequency(0.0, 1.0, 1.0)
f_big = df.backbone_frequency(1.0, 1.0, 1.0)
check("backbone frequency = 1 at zero amplitude", abs(f0 - 1.0) < 1e-9)
check("hardening resonance shifts up with amplitude", f_big > f0)
check("backbone = sqrt(alpha + 3/4 beta A^2)",
      abs(f_big - math.sqrt(1.0 + 0.75 * 1.0)) < 1e-9)
# Softening spring shifts down.
check("softening resonance shifts down",
      df.backbone_frequency(0.5, 1.0, -0.5) < df.backbone_frequency(0.0, 1.0, -0.5))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all duffing tests passed")
