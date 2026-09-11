"""Tests for kalman: matrix helpers, filtering beats measurement, smoother beats filter."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kalman import (KalmanFilter, rts_smoother, constant_velocity,
                    inverse, matmul, transpose, matvec, eye)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- matrix helpers --------------------------------------------------------
A = [[4.0, 3.0], [6.0, 3.0]]
prod = matmul(A, inverse(A))
check("inverse gives identity", all(approx(prod[i][j], 1.0 if i == j else 0.0, 1e-9)
                                    for i in range(2) for j in range(2)))
check("transpose swaps indices", transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]])
check("matvec basic", matvec([[1, 0], [0, 2]], [3, 5]) == [3, 10])
check("identity matmul", matmul(eye(2), A) == A)

# singular matrix raises
try:
    inverse([[1.0, 2.0], [2.0, 4.0]])
    check("singular matrix raises", False)
except ValueError:
    check("singular matrix raises", True)

# --- data: constant-velocity truth + Gaussian measurement noise ------------
state = 5


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def gauss(sd):
    u1 = max(1e-9, rng())
    u2 = rng()
    return sd * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


dt = 1.0
vel = 2.0
meas_sd = 5.0
T = 80
truth = [vel * t for t in range(T)]
meas = [[truth[t] + gauss(meas_sd)] for t in range(T)]


def rmse(est, lo=10):
    return math.sqrt(sum((est[t] - truth[t]) ** 2 for t in range(lo, T)) / (T - lo))


kf = constant_velocity(dt, process_var=0.01, meas_var=meas_sd ** 2, x0=[0.0, 0.0])
means, covs, pri_means, pri_covs = kf.filter(meas)

# --- filtering reduces error below the raw measurements --------------------
raw_err = rmse([m[0] for m in meas])
filt_err = rmse([m[0] for m in means])
check("filter beats raw measurements", filt_err < raw_err)
check("filter error is meaningfully lower", filt_err < 0.75 * raw_err)

# --- filtered variance falls below measurement variance --------------------
check("posterior variance < measurement variance", covs[-1][0][0] < meas_sd ** 2)
# covariance is symmetric positive on the diagonal
check("covariance diagonal positive", all(covs[-1][i][i] > 0 for i in range(2)))
check("covariance symmetric", approx(covs[-1][0][1], covs[-1][1][0], 1e-9))

# --- estimated velocity converges to the truth -----------------------------
check("recovers true velocity", approx(means[-1][1], vel, 0.3))

# --- innovation covariance shrinks then reaches steady state ---------------
# variance should be non-increasing after burn-in and converge
late = [covs[t][0][0] for t in range(40, T)]
check("variance reaches steady state", max(late) - min(late) < 0.1)

# --- RTS smoother improves on the filter -----------------------------------
kf2 = constant_velocity(dt, process_var=0.01, meas_var=meas_sd ** 2, x0=[0.0, 0.0])
sm_means, sm_covs = rts_smoother(kf2, meas)
sm_err = rmse([m[0] for m in sm_means])
check("smoother no worse than filter", sm_err <= filt_err + 1e-6)
check("smoother strictly helps here", sm_err < filt_err)
# smoothed covariance <= filtered covariance (more info used)
check("smoothed variance <= filtered", sm_covs[T // 2][0][0] <= covs[T // 2][0][0] + 1e-9)

# --- a zero-noise sensor is trusted exactly --------------------------------
# R -> 0 means the update should snap the position estimate onto the measurement
F = [[1.0]]
H = [[1.0]]
Q = [[1.0]]
R = [[1e-12]]
kf3 = KalmanFilter(F, H, Q, R, [0.0], [[1.0]])
kf3.predict()
x, P, K = kf3.update([42.0])
check("perfect sensor trusted", approx(x[0], 42.0, 1e-4))
check("perfect sensor gain ~1", approx(K[0][0], 1.0, 1e-4))

# --- a useless (infinite-noise) sensor is ignored --------------------------
kf4 = KalmanFilter([[1.0]], [[1.0]], [[0.0]], [[1e12]], [7.0], [[1.0]])
kf4.predict()
x4, P4, K4 = kf4.update([999.0])
check("useless sensor ignored", approx(x4[0], 7.0, 1e-3))

# --- static estimation: repeated noisy reads of a constant shrink variance --
kf5 = KalmanFilter([[1.0]], [[1.0]], [[0.0]], [[4.0]], [0.0], [[100.0]])
vars_seen = []
for _ in range(20):
    kf5.predict()
    kf5.update([10.0])
    vars_seen.append(kf5.P[0][0])
check("variance monotone decreasing for static", all(vars_seen[i + 1] <= vars_seen[i] + 1e-12
                                                     for i in range(len(vars_seen) - 1)))
check("static estimate converges to reading", approx(kf5.x[0], 10.0, 0.05))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all kalman tests passed")
