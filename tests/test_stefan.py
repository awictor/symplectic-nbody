"""Tests for stefan: the moving melt/freeze front."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import stefan as st

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Ice properties: k=2.2, rho=917, cp=2100, L=334000; surface 10 K below freezing.
K, RHO, CP, L = 2.2, 917.0, 2100.0, 334000.0
alpha = st.thermal_diffusivity(K, RHO, CP)
check("ice thermal diffusivity ~1.1e-6", 1.0e-6 < alpha < 1.3e-6)

St = st.stefan_number(CP, 10.0, L)
check("ice Stefan number small (~0.06)", 0.05 < St < 0.07)

lam = st.solve_lambda(St)
# For small St, lambda ~ sqrt(St/2).
check("lambda ~ sqrt(St/2) for small St", abs(lam - math.sqrt(St / 2.0)) < 0.02)

# Verify the Stefan condition is actually satisfied at the solved lambda.
lhs = lam * math.exp(lam * lam) * math.erf(lam)
check("Stefan condition satisfied", abs(lhs - St / math.sqrt(math.pi)) < 1e-6)

# One day of hard frost grows ~10 cm of ice (classic Stefan estimate, order of magnitude).
X_day = st.front_position(86400.0, lam, alpha)
check("~10 cm ice after a day of hard frost", 0.05 < X_day < 0.15)

# sqrt(t) advance: quadruple the time -> double the depth.
X1 = st.front_position(1000.0, lam, alpha)
X4 = st.front_position(4000.0, lam, alpha)
check("front doubles when time quadruples", abs(X4 - 2.0 * X1) < 1e-9)

# time_to_depth inverts front_position.
t_back = st.time_to_depth(X_day, lam, alpha)
check("time_to_depth inverts front_position", abs(t_back - 86400.0) < 1.0)

# depth^2 scaling: twice as deep takes four times as long.
check("twice the depth takes 4x the time",
      abs(st.time_to_depth(0.2, lam, alpha) - 4.0 * st.time_to_depth(0.1, lam, alpha)) < 1e-6)

# Front speed falls as 1/sqrt(t): quadruple time -> half speed.
v1 = st.front_speed(1000.0, lam, alpha)
v4 = st.front_speed(4000.0, lam, alpha)
check("front speed halves when time quadruples", abs(v4 - 0.5 * v1) < 1e-12)

# Larger Stefan number (hotter surface / less latent heat) -> larger lambda -> faster front.
lam_hot = st.solve_lambda(st.stefan_number(CP, 40.0, L))
check("hotter surface gives larger lambda", lam_hot > lam)

# Big St limit: lambda grows well past the small-St estimate.
lam_big = st.solve_lambda(10.0)
check("large St gives lambda > 1", lam_big > 1.0)

# Latent heat matters: with zero latent heat (St -> inf) the front is much faster than with
# a huge latent heat (St -> 0).
lam_lowL = st.solve_lambda(st.stefan_number(CP, 10.0, 1e4))    # small L -> big St
lam_highL = st.solve_lambda(st.stefan_number(CP, 10.0, 1e7))   # big L -> tiny St
check("smaller latent heat -> faster front", lam_lowL > lam_highL)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all stefan tests passed")
