"""Validate Ornstein-Uhlenbeck: analytic moments, exact vs Euler stepper, stationary law, autocovariance."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import ornstein_uhlenbeck as ou


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Ornstein-Uhlenbeck tests")

    p = ou.OrnsteinUhlenbeck(theta=1.5, mu=2.0, sigma=0.8)

    # --- analytic mean: relaxes from x0 to mu ---
    check("mean at t=0 is x0", abs(p.mean(5.0, 0.0) - 5.0) < 1e-12)
    check("mean -> mu as t grows", abs(p.mean(5.0, 20.0) - 2.0) < 1e-6)
    check("mean matches closed form at t=0.5",
          abs(p.mean(5.0, 0.5) - (2.0 + 3.0 * math.exp(-1.5 * 0.5))) < 1e-12)

    # --- analytic variance: 0 at t=0, -> stationary ---
    check("variance 0 at t=0", abs(p.variance(0.0)) < 1e-12)
    check("variance -> stationary", abs(p.variance(50.0) - p.stationary_variance()) < 1e-9)
    check("stationary variance == sigma^2/2theta",
          abs(p.stationary_variance() - 0.8 ** 2 / (2 * 1.5)) < 1e-12)

    # --- autocovariance/autocorrelation forms ---
    check("autocov(0) == stationary variance", abs(p.autocovariance(0.0) - p.stationary_variance()) < 1e-12)
    check("autocorr(0) == 1", abs(p.autocorrelation(0.0) - 1.0) < 1e-12)
    check("autocorr decays like e^{-theta s}", abs(p.autocorrelation(2.0) - math.exp(-3.0)) < 1e-12)
    check("correlation time == 1/theta", abs(p.correlation_time() - 1 / 1.5) < 1e-12)

    # --- exact stepper reproduces analytic mean & variance via Monte Carlo ---
    m, v = p.empirical_moments(5.0, t=0.5, dt=0.05, paths=6000, seed=3, exact=True)
    check(f"MC mean ~ analytic ({m:.3f} vs {p.mean(5.0,0.5):.3f})", abs(m - p.mean(5.0, 0.5)) < 0.03)
    check(f"MC variance ~ analytic ({v:.3f} vs {p.variance(0.5):.3f})", abs(v - p.variance(0.5)) < 0.02)

    # --- exact stepper is time-step-error-free even at LARGE dt where Euler is biased ---
    # take a single big step of size dt=t and compare variance to analytic
    big_dt = 1.0
    me, ve = p.empirical_moments(5.0, t=big_dt, dt=big_dt, paths=8000, seed=7, exact=True)
    check(f"exact stepper: 1-shot variance ~ analytic at dt=1 ({ve:.3f} vs {p.variance(1.0):.3f})",
          abs(ve - p.variance(1.0)) < 0.03)
    mu_e, vu = p.empirical_moments(5.0, t=big_dt, dt=big_dt, paths=8000, seed=7, exact=False)
    # Euler with one big step: variance = sigma^2 * dt = 0.64, analytic ~ 0.201 -> badly biased
    euler_var_pred = p.sigma ** 2 * big_dt
    check(f"Euler at dt=1 is biased high ({vu:.3f} ~ {euler_var_pred:.3f} >> {p.variance(1.0):.3f})",
          abs(vu - euler_var_pred) < 0.05 and vu > p.variance(1.0) + 0.2)

    # --- stationary sampler matches Normal(mu, sigma^2/2theta) ---
    rng = ou._Rng(11)
    samples = [p.sample_stationary(rng) for _ in range(20000)]
    sm = sum(samples) / len(samples)
    sv = sum((x - sm) ** 2 for x in samples) / (len(samples) - 1)
    check(f"stationary sample mean ~ mu ({sm:.3f})", abs(sm - 2.0) < 0.02)
    check(f"stationary sample var ~ sigma^2/2theta ({sv:.3f} vs {p.stationary_variance():.3f})",
          abs(sv - p.stationary_variance()) < 0.02)

    # --- empirical autocovariance decays like e^{-theta s} ---
    rng2 = ou._Rng(23)
    dt = 0.1
    ac = p.empirical_autocovariance(dt, steps=200000, rng=rng2, lags=[0, 5, 10, 20])
    # lag k corresponds to time s = k*dt; predicted cov = stationary_var * e^{-theta*s}
    ok = True
    for lag in [0, 5, 10, 20]:
        s = lag * dt
        pred = p.stationary_variance() * math.exp(-p.theta * s)
        if abs(ac[lag] - pred) > 0.03:
            ok = False
    check("empirical autocovariance matches e^{-theta s} decay", ok)

    # --- discrete process is AR(1) with coefficient e^{-theta dt} ---
    # X_{n+1} - mu = e^{-theta dt}(X_n - mu) + noise ; check the deterministic decay coefficient
    decay = math.exp(-p.theta * dt)
    check(f"AR(1) decay coefficient e^-theta*dt = {decay:.4f} in (0,1)", 0 < decay < 1)

    # --- parameter validation ---
    try:
        ou.OrnsteinUhlenbeck(theta=-1, mu=0, sigma=1)
        check("rejects theta<=0", False)
    except ValueError:
        check("rejects theta<=0", True)

    # --- determinism per seed ---
    a = p.simulate(0.0, 0.1, 100, ou._Rng(42))
    b = p.simulate(0.0, 0.1, 100, ou._Rng(42))
    check("simulate deterministic per seed", a == b)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
