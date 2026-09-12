"""Tests for rk45: known solutions, tolerance control, adaptive stepping, energy conservation."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rk45 import solve, solve_at

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- exponential decay y' = -y, y(0)=1 -> e^-t -----------------------------
ts, ys = solve(lambda t, y: -y, 0.0, 1.0, 5.0, atol=1e-9, rtol=1e-9)
check("exponential decay final value accurate", abs(ys[-1] - math.exp(-5)) < 1e-6)
check("solution starts at the initial condition", ys[0] == 1.0)
check("times start at t0 and end at t1", ts[0] == 0.0 and abs(ts[-1] - 5.0) < 1e-12)

# --- exponential growth y' = y -> e^t --------------------------------------
ts, ys = solve(lambda t, y: y, 0.0, 1.0, 3.0, atol=1e-10, rtol=1e-10)
check("exponential growth accurate", abs(ys[-1] - math.exp(3)) / math.exp(3) < 1e-7)

# --- tightening the tolerance reduces the error ----------------------------
def decay_error(tol):
    _, ys = solve(lambda t, y: -y, 0.0, 1.0, 5.0, atol=tol, rtol=tol)
    return abs(ys[-1] - math.exp(-5))

e_loose = decay_error(1e-4)
e_tight = decay_error(1e-10)
check(f"tighter tolerance gives smaller error ({e_loose:.2e} -> {e_tight:.2e})", e_tight < e_loose)
check("loose-tolerance error respects the tolerance roughly", e_loose < 1e-2)

# --- simple harmonic motion: y'' = -y, conserves energy --------------------
# state [x, v]; energy = 0.5(x^2 + v^2) should stay constant
ts, ys = solve(lambda t, y: [y[1], -y[0]], 0.0, [1.0, 0.0], 20 * math.pi,
               atol=1e-10, rtol=1e-10)
energies = [0.5 * (s[0] ** 2 + s[1] ** 2) for s in ys]
drift = max(abs(e - 0.5) for e in energies)
check(f"SHM conserves energy over 10 periods (drift {drift:.2e})", drift < 1e-6)
# after a whole number of periods, return to the start
check("SHM returns to the start after 10 full periods",
      abs(ys[-1][0] - 1.0) < 1e-5 and abs(ys[-1][1]) < 1e-5)

# --- SHM position matches cos(t) at sampled times --------------------------
times = [i * math.pi / 4 for i in range(9)]
sol = solve_at(lambda t, y: [y[1], -y[0]], 0.0, [1.0, 0.0], times, atol=1e-10, rtol=1e-10)
# solve_at uses linear interpolation between steps, so allow a looser bound
ok = all(abs(sol[i][0] - math.cos(times[i])) < 1e-3 for i in range(len(times)))
check("sampled SHM matches cos(t) at requested times", ok)

# --- logistic growth y' = y(1-y) -> 1/(1+((1-y0)/y0)e^-t) ------------------
y0 = 0.1
ts, ys = solve(lambda t, y: y * (1 - y), 0.0, y0, 10.0, atol=1e-10, rtol=1e-10)
exact = 1 / (1 + ((1 - y0) / y0) * math.exp(-10))
check("logistic curve final value accurate", abs(ys[-1] - exact) < 1e-7)
check("logistic solution stays in (0,1)", all(0 < v < 1.0001 for v in ys))

# --- adaptive stepping: more steps where the solution changes fastest ------
# a solution with a sharp transition: y' = -50 y around a fast decay uses tighter steps
# compare step counts for a fast vs slow decay over the same interval
_, slow = solve(lambda t, y: -1.0 * y, 0.0, 1.0, 5.0, atol=1e-8, rtol=1e-8)
ts_fast, _ = solve(lambda t, y: -20.0 * y, 0.0, 1.0, 5.0, atol=1e-8, rtol=1e-8)
ts_slow, _ = solve(lambda t, y: -1.0 * y, 0.0, 1.0, 5.0, atol=1e-8, rtol=1e-8)
check("faster dynamics require more steps (adaptive control)", len(ts_fast) > len(ts_slow))

# --- a small linear system y' = A y ----------------------------------------
# y1' = y2, y2' = -4 y1 -> oscillation with frequency 2; y1 = cos(2t)
ts, ys = solve(lambda t, y: [y[1], -4 * y[0]], 0.0, [1.0, 0.0], math.pi,
               atol=1e-10, rtol=1e-10)
check("2-frequency oscillator matches cos(2t) at t=pi", abs(ys[-1][0] - math.cos(2 * math.pi)) < 1e-6)

# --- integrating backward in time -----------------------------------------
ts, ys = solve(lambda t, y: -y, 0.0, 1.0, -2.0, atol=1e-9, rtol=1e-9)
check("backward integration works (y' = -y from 0 to -2)", abs(ys[-1] - math.exp(2)) / math.exp(2) < 1e-6)

# --- zero-derivative (constant solution) -----------------------------------
ts, ys = solve(lambda t, y: 0.0, 0.0, 7.0, 3.0)
check("constant solution stays constant", all(abs(v - 7.0) < 1e-12 for v in ys))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all rk45 tests passed")
