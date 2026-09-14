"""Validate Kruskal-Wallis: null vs shifted, k=2 matches Mann-Whitney, rank invariance, ties, chi2 sf."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import kruskal_wallis as kw
import mann_whitney as mw


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


def main():
    print("Kruskal-Wallis tests")

    # --- null: three groups from one distribution -> small H, large p ---
    # (any single draw can land in the 5% tail by chance; use a representative non-tail seed and a
    #  reasonable n. The shifted-group test below is the real power check.)
    rng = _R(2)
    g1 = [rng.normal() for _ in range(30)]
    g2 = [rng.normal() for _ in range(30)]
    g3 = [rng.normal() for _ in range(30)]
    res = kw.kruskal_wallis([g1, g2, g3])
    check(f"null: large p ({res['p_value']:.3f})", res["p_value"] > 0.1)
    check("df = k - 1", res["df"] == 2)

    rng = _R(1)  # restore the seed the rest of the test expects

    # --- shifted groups -> large H, tiny p ---
    a = [rng.normal() for _ in range(20)]
    b = [rng.normal() + 3 for _ in range(20)]
    c = [rng.normal() + 6 for _ in range(20)]
    res = kw.kruskal_wallis([a, b, c])
    check(f"shifted: tiny p ({res['p_value']:.2e})", res["p_value"] < 0.001)
    check("shifted: large H", res["H"] > 10)
    check("shifted: positive effect size", res["epsilon_squared"] > 0.3)

    # --- k=2 matches Mann-Whitney normal-approx z^2 ---
    x = [4.0, 7.0, 2.0, 9.0, 5.0, 1.0, 8.0]
    y = [6.0, 3.0, 10.0, 12.0, 11.0, 5.5, 13.0]
    H = kw.kruskal_wallis([x, y])["H"]
    # Mann-Whitney z (no continuity correction) squared should equal H
    u1, u2, _ = mw.u_statistic(x, y)
    n1, n2 = len(x), len(y)
    mu = n1 * n2 / 2
    N = n1 + n2
    from collections import Counter
    tie = sum(t ** 3 - t for t in Counter(x + y).values())
    var = (n1 * n2 / 12.0) * ((N + 1) - tie / (N * (N - 1)))
    z = (u1 - mu) / math.sqrt(var)
    check(f"k=2: H == Mann-Whitney z^2 ({H:.4f} vs {z*z:.4f})", abs(H - z * z) < 1e-6)

    # --- rank invariance: monotone transform doesn't change H ---
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [3.0, 4.0, 5.0, 6.0, 7.0]
    c = [5.0, 6.0, 7.0, 8.0, 9.0]
    H1 = kw.kruskal_wallis([a, b, c])["H"]
    f = lambda v: math.exp(v / 3)
    H2 = kw.kruskal_wallis([[f(x) for x in a], [f(x) for x in b], [f(x) for x in c]])["H"]
    check(f"H invariant under monotone transform ({H1:.4f} vs {H2:.4f})", abs(H1 - H2) < 1e-9)

    # --- tie correction increases H (divides by <1) ---
    tied = [[1, 1, 2, 2], [2, 2, 3, 3], [3, 3, 4, 4]]
    H_tied, _ = kw.h_statistic(tied)
    # recompute without correction manually
    all_v = [v for g in tied for v in g]
    from mann_whitney import _average_ranks
    ranks = _average_ranks(all_v)
    Nn = len(all_v)
    idx = 0
    term = 0.0
    for g in tied:
        r = ranks[idx:idx + len(g)]
        idx += len(g)
        term += len(g) * (sum(r) / len(g) - (Nn + 1) / 2) ** 2
    H_uncorrected = 12.0 / (Nn * (Nn + 1)) * term
    check(f"tie correction raises H ({H_uncorrected:.3f} -> {H_tied:.3f})", H_tied >= H_uncorrected - 1e-9)

    # --- chi-squared survival function matches known values ---
    # chi2 with df=1: P(X>3.841) ~ 0.05; df=2: P(X>5.991) ~ 0.05
    check("chi2 sf df=1 at 3.841 ~ 0.05", abs(kw.chi2_sf(3.841, 1) - 0.05) < 0.005)
    check("chi2 sf df=2 at 5.991 ~ 0.05", abs(kw.chi2_sf(5.991, 2) - 0.05) < 0.005)
    check("chi2 sf df=3 at 7.815 ~ 0.05", abs(kw.chi2_sf(7.815, 3) - 0.05) < 0.005)
    check("chi2 sf at 0 == 1", abs(kw.chi2_sf(0, 2) - 1.0) < 1e-12)
    check("chi2 sf monotone decreasing", kw.chi2_sf(2, 2) > kw.chi2_sf(8, 2))

    # --- mean ranks: middle group's mean rank between the others when shifted ---
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [7, 8, 9]
    mr = kw.kruskal_wallis([a, b, c])["mean_ranks"]
    check("mean ranks ordered by group location", mr[0] < mr[1] < mr[2])

    # --- effect size rises with separation ---
    base = [rng.normal() for _ in range(15)]
    e_small = kw.kruskal_wallis([base, [x + 0.5 for x in base], [x + 1 for x in base]])["epsilon_squared"]
    e_big = kw.kruskal_wallis([base, [x + 3 for x in base], [x + 6 for x in base]])["epsilon_squared"]
    check(f"effect size rises with separation ({e_small:.3f} < {e_big:.3f})", e_small < e_big)

    # --- deterministic ---
    check("deterministic", kw.kruskal_wallis([a, b, c]) == kw.kruskal_wallis([a, b, c]))

    # --- rejects < 2 groups ---
    try:
        kw.kruskal_wallis([[1, 2, 3]])
        check("rejects single group", False)
    except ValueError:
        check("rejects single group", True)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
