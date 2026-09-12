"""Tests for tdigest: quantiles vs exact-sorted, sharper tails, bounded size, merge correctness."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tdigest import TDigest, merge_digests  # noqa: E402


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self):
        return sum(self.u() for _ in range(12)) - 6.0

    def expo(self, lam=1.0):
        u = self.u()
        return -math.log(1 - u) / lam


def exact_quantile(sorted_data, q):
    if q <= 0:
        return sorted_data[0]
    if q >= 1:
        return sorted_data[-1]
    idx = q * (len(sorted_data) - 1)
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def rank_error(td, sorted_data, q):
    """Absolute error in RANK space: |cdf(estimate) - q| computed against the true data."""
    est = td.quantile(q)
    # true rank of est
    lo, hi = 0, len(sorted_data)
    import bisect
    r = bisect.bisect_left(sorted_data, est) / len(sorted_data)
    return abs(r - q)


def main():
    rng = LCG(2024)

    # ---- 1. uniform stream: quantiles close to exact ----------------------------------
    data = [rng.u() * 1000 for _ in range(40000)]
    td = TDigest(delta=100)
    td.add_all(data)
    sd = sorted(data)
    for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        est = td.quantile(q)
        ex = exact_quantile(sd, q)
        rng_span = sd[-1] - sd[0]
        check(f"uniform quantile q={q} close to exact", abs(est - ex) < 0.02 * rng_span,
              f"est={est:.2f} exact={ex:.2f}")

    # ---- 2. tails are sharper than the median (the whole point) -----------------------
    # compare rank error at p50 vs p99 -- p99 should be no worse, typically better
    e50 = rank_error(td, sd, 0.5)
    e99 = rank_error(td, sd, 0.99)
    e999 = rank_error(td, sd, 0.999)
    check("tail rank error (p99) <= median rank error + slack", e99 <= e50 + 0.01,
          f"e50={e50:.4f} e99={e99:.4f}")
    check("deep tail (p999) rank error small", e999 < 0.01, f"e999={e999:.4f}")

    # ---- 3. normal and exponential streams --------------------------------------------
    for name, gen in (("normal", lambda: rng.normal() * 10 + 50),
                      ("exponential", lambda: rng.expo(0.5))):
        d = [gen() for _ in range(30000)]
        t = TDigest(100)
        t.add_all(d)
        s = sorted(d)
        worst = max(abs(t.quantile(q) - exact_quantile(s, q)) / (s[-1] - s[0])
                    for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
        check(f"{name} quantiles within tolerance", worst < 0.03, f"worst rel err {worst:.4f}")

    # ---- 4. centroid count stays bounded ----------------------------------------------
    big = TDigest(100)
    for _ in range(200000):
        big.add(rng.u())
    cc = big.centroid_count()
    check("centroid count bounded by ~small multiple of delta", cc <= 100 * 3,
          f"{cc} centroids for 200k values")

    # ---- 5. merge correctness ---------------------------------------------------------
    whole = [rng.normal() for _ in range(60000)]
    single = TDigest(100)
    single.add_all(whole)
    # split across 6 shards
    shards = [TDigest(100) for _ in range(6)]
    for i, v in enumerate(whole):
        shards[i % 6].add(v)
    merged = merge_digests(shards, 100)
    sw = sorted(whole)
    for q in (0.1, 0.5, 0.9, 0.99):
        em = merged.quantile(q)
        es = single.quantile(q)
        ex = exact_quantile(sw, q)
        span = sw[-1] - sw[0]
        check(f"merged digest quantile q={q} matches single + exact",
              abs(em - ex) < 0.03 * span and abs(em - es) < 0.03 * span,
              f"merged={em:.3f} single={es:.3f} exact={ex:.3f}")

    # ---- 6. quantile/CDF properties ---------------------------------------------------
    t = TDigest(100)
    t.add_all([rng.u() * 100 for _ in range(20000)])
    check("quantile(0) == min", abs(t.quantile(0.0) - t.min) < 1e-9)
    check("quantile(1) == max", abs(t.quantile(1.0) - t.max) < 1e-9)
    # monotone quantile
    qs = [t.quantile(q) for q in [i / 20 for i in range(21)]]
    check("quantile is monotone non-decreasing", all(qs[i] <= qs[i + 1] + 1e-9 for i in range(20)))
    # monotone CDF
    xs = [i * 5 for i in range(21)]
    cdfs = [t.cdf(x) for x in xs]
    check("cdf is monotone non-decreasing", all(cdfs[i] <= cdfs[i + 1] + 1e-9 for i in range(20)))
    check("cdf in [0,1]", all(0 <= c <= 1 for c in cdfs))
    # quantile/cdf approximate inverse
    mid = t.quantile(0.5)
    check("cdf(quantile(0.5)) ~ 0.5", abs(t.cdf(mid) - 0.5) < 0.05, f"{t.cdf(mid):.4f}")

    # ---- 7. order robustness ----------------------------------------------------------
    vals = [rng.normal() for _ in range(20000)]
    t1 = TDigest(100); t1.add_all(vals)
    shuffled = vals[::-1]
    t2 = TDigest(100); t2.add_all(shuffled)
    span = max(vals) - min(vals)
    check("shuffled input gives similar median", abs(t1.quantile(0.5) - t2.quantile(0.5)) < 0.02 * span)

    # ---- 8. weighted add --------------------------------------------------------------
    tw = TDigest(100)
    tw.add(10.0, weight=1000)
    tw.add(20.0, weight=1000)
    med = tw.quantile(0.5)
    check("weighted values respected", 10.0 <= med <= 20.0, f"median={med}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
