"""Tests for particle_filter: ESS, systematic resampling, nonlinear tracking, degeneracy."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from particle_filter import (ParticleFilter, effective_sample_size, systematic_resample,
                             gaussian_likelihood, _Rng)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- effective sample size -------------------------------------------------
check("ESS uniform = N", approx(effective_sample_size([0.25] * 4), 4.0, 1e-9))
check("ESS degenerate = 1", approx(effective_sample_size([1.0, 0.0, 0.0, 0.0]), 1.0, 1e-9))
check("ESS two-equal = 2", approx(effective_sample_size([0.5, 0.5, 0.0, 0.0]), 2.0, 1e-9))

# --- systematic resampling favours heavy weights ---------------------------
rng = _Rng(1)
idx = systematic_resample([0.7, 0.1, 0.1, 0.1], rng)
check("resample duplicates the heavy particle", idx.count(0) >= 2)
check("resample returns N indices", len(idx) == 4)
check("resample indices in range", all(0 <= i < 4 for i in idx))
# a uniform-weight resample keeps roughly one of each
uni_idx = systematic_resample([0.25] * 4, _Rng(5))
check("uniform resample covers all", len(set(uni_idx)) == 4)

# --- Gaussian likelihood ---------------------------------------------------
check("likelihood peaks at match", gaussian_likelihood(3.0, 3.0, 1.0) == 1.0)
check("likelihood decays with error",
      gaussian_likelihood(0.0, 1.0, 1.0) > gaussian_likelihood(0.0, 2.0, 1.0) > 0)

# --- Gaussian sampler is roughly standard normal ---------------------------
r = _Rng(11)
samples = [r.gauss(0.0, 1.0) for _ in range(4000)]
mean = sum(samples) / len(samples)
var = sum((s - mean) ** 2 for s in samples) / len(samples)
check("sampler mean ~ 0", abs(mean) < 0.1)
check("sampler variance ~ 1", abs(var - 1.0) < 0.15)

# --- nonlinear tracking: x_{t+1} = x + 0.3 sin(x) + 0.5 + noise, z = x + noise
def trans(x, rng):
    return x + 0.3 * math.sin(x) + 0.5 + rng.gauss(0.0, 0.3)


def lik(x, z):
    return gaussian_likelihood(x, z, 1.0)


def init(rng):
    return rng.gauss(0.0, 1.0)


gen = _Rng(7)
x = 0.0
truth, meas = [], []
for _ in range(40):
    x = x + 0.3 * math.sin(x) + 0.5 + gen.gauss(0.0, 0.3)
    truth.append(x)
    meas.append(x + gen.gauss(0.0, 1.0))


def track_err(est):
    return sum(abs(est[t] - truth[t]) for t in range(5, 40)) / 35


pf = ParticleFilter(trans, lik, init, n_particles=400, seed=3)
est = pf.run(pf_meas := meas)
raw_err = sum(abs(meas[t] - truth[t]) for t in range(5, 40)) / 35
check("PF estimate beats the raw measurements", track_err(est) < raw_err)
check("PF tracking error is small", track_err(est) < 0.5)
check("PF returns one estimate per step", len(est) == 40)

# --- adaptive resampling keeps ESS healthy; no resampling collapses it -----
pf_nr = ParticleFilter(trans, lik, init, n_particles=400, resample_threshold=0.0, seed=3)
pf_nr.run(meas)
check("without resampling the ESS collapses", min(pf_nr.ess_history) < 20)
check("with adaptive resampling the ESS stays high", min(pf.ess_history) > 50)
check("adaptive filter did resample at least once", any(pf.resampled_history))

# --- more particles reduce error (averaged over seeds) ---------------------
def avg_err(n):
    errs = []
    for s in range(3):
        p = ParticleFilter(trans, lik, init, n_particles=n, seed=s)
        errs.append(track_err(p.run(meas)))
    return sum(errs) / len(errs)


check("more particles reduce error", avg_err(500) <= avg_err(25) + 1e-9)

# --- a 2-D state (position, velocity) tracked from position-only obs --------
def trans2(s, rng):
    px, vx = s
    return [px + vx + rng.gauss(0.0, 0.05), vx + rng.gauss(0.0, 0.05)]


def lik2(s, z):
    return gaussian_likelihood(s[0], z, 0.5)


def init2(rng):
    return [rng.gauss(0.0, 1.0), rng.gauss(0.0, 0.5)]


g2 = _Rng(21)
pos, vel = 0.0, 0.4
truth2, meas2 = [], []
for _ in range(30):
    pos += vel
    truth2.append(pos)
    meas2.append(pos + g2.gauss(0.0, 0.5))
pf2 = ParticleFilter(trans2, lik2, init2, n_particles=500, seed=4)
est2 = pf2.run(meas2)
pos_err = sum(abs(est2[t][0] - truth2[t]) for t in range(5, 30)) / 25
check("2-D PF tracks position", pos_err < 0.5)
check("2-D PF infers a positive velocity", est2[-1][1] > 0.1)
check("2-D estimate is a 2-vector", len(est2[0]) == 2)

# --- determinism -----------------------------------------------------------
a = ParticleFilter(trans, lik, init, n_particles=100, seed=9).run(meas)
b = ParticleFilter(trans, lik, init, n_particles=100, seed=9).run(meas)
check("particle filter deterministic for a fixed seed", a == b)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all particle_filter tests passed")
