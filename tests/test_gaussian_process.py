"""Tests for gaussian_process: interpolation, calibrated variance, marginal likelihood."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gaussian_process import GaussianProcess, fit_length_scale, rbf_kernel

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- the RBF kernel ---------------------------------------------------------
check("kernel peaks at zero distance", approx(rbf_kernel(2.0, 2.0), 1.0, 1e-12))
check("kernel decays with distance", rbf_kernel(0.0, 1.0) > rbf_kernel(0.0, 2.0) > 0)
check("kernel is symmetric", approx(rbf_kernel(1.0, 3.0), rbf_kernel(3.0, 1.0), 1e-12))
check("signal variance scales kernel", approx(rbf_kernel(0.0, 0.0, signal_var=4.0), 4.0, 1e-12))
# shorter length scale => faster decay
check("short length scale decays faster",
      rbf_kernel(0.0, 1.0, length_scale=0.5) < rbf_kernel(0.0, 1.0, length_scale=2.0))
# 2-D input works
check("kernel handles vector inputs", approx(rbf_kernel([0.0, 0.0], [0.0, 0.0]), 1.0, 1e-12))

# --- noise-free interpolation ----------------------------------------------
Xtr = [0.0, 1.0, 2.0, 3.0, 4.0]
ytr = [math.sin(x) for x in Xtr]
gp = GaussianProcess(length_scale=1.0, noise_var=1e-10).fit(Xtr, ytr)
means, stds = gp.predict(Xtr)
check("interpolates training points", max(abs(means[i] - ytr[i]) for i in range(5)) < 1e-4)
check("zero variance at training points", max(stds) < 1e-3)

# --- recovers a smooth function between training points --------------------
qx = [0.5, 1.5, 2.5, 3.5]
qm, qs = gp.predict(qx)
truth = [math.sin(x) for x in qx]
check("recovers sin between points", max(abs(qm[i] - truth[i]) for i in range(4)) < 0.1)
check("interior std positive but small", all(0.0 < s < 0.5 for s in qs))

# --- uncertainty grows away from the data ----------------------------------
m2, s2 = gp.predict([2.0, 7.0, 20.0])   # inside, just outside, far outside
check("uncertainty grows with distance from data", s2[0] < s2[1] < s2[2])

# --- far from data, posterior reverts to the prior -------------------------
mf, sf = gp.predict([50.0])
check("mean reverts to prior mean 0", abs(mf[0]) < 1e-3)
check("std reverts to prior signal std", approx(sf[0], 1.0, 1e-3))

# --- variance is never negative (numerically) ------------------------------
grid = [x * 0.13 for x in range(60)]
_, gstd = gp.predict(grid)
check("all standard deviations real & nonnegative", all(s >= 0.0 for s in gstd))

# --- observation noise makes the fit smooth (not exact interpolation) ------
gp_noisy = GaussianProcess(length_scale=1.0, noise_var=0.25).fit(Xtr, ytr)
mn, sn = gp_noisy.predict(Xtr)
check("noise => nonzero variance at training points", min(sn) > 0.0)
check("noisy fit stays close to data", max(abs(mn[i] - ytr[i]) for i in range(5)) < 0.5)

# --- log marginal likelihood is finite and hyperparameters are selectable --
lml = gp.log_marginal_likelihood()
check("log marginal likelihood is finite", math.isfinite(lml))

# marginal likelihood prefers a LONGER length scale for a smooth function and
# a SHORTER one for a wiggly function
gridls = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0]
Xd = [i * 0.3 for i in range(20)]
smooth = [math.sin(0.4 * x) for x in Xd]
rough = [math.sin(3.0 * x) for x in Xd]
_, ls_smooth, _ = fit_length_scale(Xd, smooth, gridls, noise_var=1e-6)
_, ls_rough, _ = fit_length_scale(Xd, rough, gridls, noise_var=1e-6)
check("smooth function prefers longer length scale", ls_smooth > ls_rough)

# --- selected model has the highest marginal likelihood in the grid --------
best_gp, best_ls, best_lml = fit_length_scale(Xd, smooth, gridls, noise_var=1e-6)
all_lmls = []
for ls in gridls:
    g = GaussianProcess(length_scale=ls, noise_var=1e-6).fit(Xd, smooth)
    all_lmls.append(g.log_marginal_likelihood())
check("selected length scale maximizes marginal likelihood", approx(best_lml, max(all_lmls), 1e-9))

# --- 2-D regression: fit f(x,y) = x + y ------------------------------------
X2 = [[a, b] for a in (0.0, 1.0, 2.0) for b in (0.0, 1.0, 2.0)]
y2 = [a + b for a, b in X2]
gp2 = GaussianProcess(length_scale=1.5, noise_var=1e-8).fit(X2, y2)
p2, _ = gp2.predict([[1.0, 1.0], [0.5, 0.5]])
check("2-D GP interpolates a grid point", approx(p2[0], 2.0, 1e-3))
check("2-D GP interpolates between points", approx(p2[1], 1.0, 0.2))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all gaussian_process tests passed")
