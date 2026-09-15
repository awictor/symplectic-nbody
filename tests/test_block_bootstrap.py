"""Validate block bootstrap: matches i.i.d. on independent data, wider SE on AR(1), coverage, variants."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import block_bootstrap as bb
import bootstrap as bs


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
        self._spare = None

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self):
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return v
        u1 = max(self.u(), 1e-12)
        u2 = self.u()
        r = math.sqrt(-2 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)


def ar1(n, phi, rng, innov=1.0):
    x = [0.0]
    for _ in range(n + 200):
        x.append(phi * x[-1] + innov * rng.normal())
    return x[201:201 + n]


def main():
    print("Block bootstrap tests")

    # --- resampled series have the right length ---
    rng = bb._Rng(1)
    data = list(range(50))
    for method, fn in (("moving", bb.moving_block_resample), ("circular", bb.circular_block_resample)):
        r = fn(data, 5, rng)
        check(f"{method} resample length == n", len(r) == len(data))
    rs = bb.stationary_resample(data, 5, rng)
    check("stationary resample length == n", len(rs) == len(data))

    # --- resampled values are all from the original data ---
    r = bb.circular_block_resample(data, 7, rng)
    check("resample values subset of data", set(r) <= set(data))

    # --- on INDEPENDENT data, block SE ~ i.i.d. bootstrap SE ---
    rng = _R(3)
    indep = [rng.normal() for _ in range(300)]
    se_block = bb.block_bootstrap_se(indep, bb.mean, block_len=5, n_resamples=800, seed=1)
    se_iid = bs.bootstrap_se(indep, bs.mean, n_resamples=800, seed=1)
    check(f"independent: block SE ~ iid SE ({se_block:.4f} vs {se_iid:.4f})",
          abs(se_block - se_iid) < 0.3 * se_iid)

    # --- on AR(1), block SE >> naive iid SE (correlation inflates true variance) ---
    rng = _R(5)
    series = ar1(400, phi=0.8, rng=rng)
    se_block = bb.block_bootstrap_se(series, bb.mean, block_len=20, n_resamples=800, seed=1)
    se_iid = bs.bootstrap_se(series, bs.mean, n_resamples=800, seed=1)
    check(f"AR(1): block SE > iid SE ({se_block:.4f} > {se_iid:.4f})", se_block > 1.5 * se_iid)

    # --- block SE approaches the analytic long-run SE for AR(1) ---
    # Var(mean) ~ sigma^2/n * (1+phi)/(1-phi) for AR(1); sigma^2 = innov^2/(1-phi^2)
    phi = 0.8
    n = 400
    sigma2 = 1.0 / (1 - phi ** 2)
    lr_var = sigma2 / n * (1 + phi) / (1 - phi)
    lr_se = math.sqrt(lr_var)
    check(f"block SE near analytic long-run SE ({se_block:.4f} vs {lr_se:.4f})",
          abs(se_block - lr_se) < 0.5 * lr_se)

    # --- coverage: block CI covers the true mean more often than iid CI on AR(1) ---
    true_mean = 0.0
    cover_block = 0
    cover_iid = 0
    trials = 40
    for t in range(trials):
        r = _R(100 + t)
        s = ar1(200, phi=0.7, rng=r)
        lo_b, hi_b = bb.block_bootstrap_ci(s, bb.mean, block_len=12, confidence=0.9,
                                           n_resamples=300, seed=t + 1)
        lo_i, hi_i = bs.percentile_ci(s, bs.mean, confidence=0.9, n_resamples=300, seed=t + 1)
        if lo_b <= true_mean <= hi_b:
            cover_block += 1
        if lo_i <= true_mean <= hi_i:
            cover_iid += 1
    check(f"block CI covers more than iid on AR(1) ({cover_block}/{trials} vs {cover_iid}/{trials})",
          cover_block > cover_iid)

    # --- stationary variant: mean block length matches parameter ---
    rng = bb._Rng(7)
    lengths = []
    for _ in range(5000):
        lengths.append(rng.geometric(1.0 / 8))
    mean_len = sum(lengths) / len(lengths)
    check(f"geometric mean block length ~ 8 ({mean_len:.2f})", abs(mean_len - 8) < 1.0)

    # --- optimal block length grows like n^(1/3) ---
    check("optimal block length rule", bb.optimal_block_length(1000) == 10)
    check("optimal block length grows", bb.optimal_block_length(8000) > bb.optimal_block_length(1000))

    # --- reproducible per seed ---
    a = bb.block_bootstrap(series, bb.mean, 20, n_resamples=100, seed=42)
    b = bb.block_bootstrap(series, bb.mean, 20, n_resamples=100, seed=42)
    check("reproducible per seed", a == b)
    c = bb.block_bootstrap(series, bb.mean, 20, n_resamples=100, seed=43)
    check("differs across seeds", a != c)

    # --- circular block uses wrap-around (block starting near the end includes early points) ---
    rng = bb._Rng(1)
    d = list(range(10))
    # force many resamples; the value 0 should sometimes follow 9 (wrap)
    found_wrap = False
    for _ in range(200):
        r = bb.circular_block_resample(d, 4, rng)
        for i in range(len(r) - 1):
            if r[i] == 9 and r[i + 1] == 0:
                found_wrap = True
                break
        if found_wrap:
            break
    check("circular block wraps around", found_wrap)

    # --- unknown method raises ---
    try:
        bb.block_bootstrap(series, bb.mean, 10, method="nope")
        check("rejects unknown method", False)
    except ValueError:
        check("rejects unknown method", True)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
