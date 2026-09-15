"""Validate Wilcoxon signed-rank: shift detection, exact vs normal, scale invariance, ties/zeros, null."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import wilcoxon_signed_rank as wsr


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
    print("Wilcoxon signed-rank tests")

    # --- consistent positive shift -> tiny p ---
    rng = _R(1)
    diffs = [1.5 + 0.3 * rng.normal() for _ in range(15)]  # all ~positive
    res = wsr.wilcoxon(diffs)
    check(f"positive shift: tiny p ({res['p_value']:.2e})", res["p_value"] < 0.01)
    check("positive shift: W- small (mostly positive)", res["W_minus"] < res["W_plus"])

    # --- symmetric noise around zero -> large p ---
    rng = _R(3)
    diffs = [rng.normal() for _ in range(20)]
    res = wsr.wilcoxon(diffs)
    check(f"null: large p ({res['p_value']:.3f})", res["p_value"] > 0.1)

    # --- W+ + W- == n(n+1)/2 ---
    wp, wm, n, _ = wsr._signed_ranks(diffs)
    check("W+ + W- == n(n+1)/2", abs(wp + wm - n * (n + 1) / 2) < 1e-9)

    # --- exact and normal p-values agree for moderate n ---
    diffs = [0.8, -0.3, 1.2, 0.5, -0.1, 0.9, 1.5, -0.6, 0.4, 1.1, 0.7, -0.2]
    pe = wsr.exact_pvalue(diffs)
    pn = wsr.normal_pvalue(diffs)
    check(f"exact ~ normal ({pe:.3f} vs {pn:.3f})", abs(pe - pn) < 0.1)

    # --- scale invariance: multiplying all differences by a constant doesn't change W or p ---
    d1 = [0.5, -0.2, 0.9, 0.3, -0.1, 0.7, 0.4]
    r1 = wsr.wilcoxon(d1)
    d2 = [10 * v for v in d1]
    r2 = wsr.wilcoxon(d2)
    check("scale-invariant W", r1["W"] == r2["W"])
    check("scale-invariant p", abs(r1["p_value"] - r2["p_value"]) < 1e-12)

    # --- outlier robustness: one wild value doesn't flip the rank test ---
    d = [0.4, 0.5, 0.6, 0.3, 0.7, 0.5, 0.4, 0.6]   # consistent positive
    r_clean = wsr.wilcoxon(d)
    d_out = d + [-100.0]   # one huge negative outlier
    r_out = wsr.wilcoxon(d_out)
    # still significant-ish: the outlier is just one rank (the largest), not a mean-swing
    check(f"robust to outlier (p {r_clean['p_value']:.3f} -> {r_out['p_value']:.3f})",
          r_out["p_value"] < 0.2)

    # --- zeros dropped ---
    wp, wm, n, _ = wsr._signed_ranks([1.0, 0.0, -2.0, 0.0, 3.0])
    check("zeros dropped from n", n == 3)

    # --- ties in |d| get average ranks ---
    wp, wm, n, ranks = wsr._signed_ranks([2.0, -2.0, 3.0])   # |2|,|2| tie -> ranks 1.5,1.5
    check("tied |d| average ranks", sorted(ranks) == [1.5, 1.5, 3.0])

    # --- tied data uses normal approx ---
    res = wsr.wilcoxon([2.0, -2.0, 3.0, 3.0, 1.0])
    check("tied data uses normal approx", res["method"] == "normal-approx")

    # --- exact null distribution symmetric and sums to 2^n ---
    counts = wsr._exact_null_counts(6)
    check("exact null sums to 2^n", sum(counts) == 2 ** 6)
    check("exact null symmetric", counts == counts[::-1])

    # --- paired-sample interface (x, y) ---
    x = [5.2, 6.1, 4.8, 7.0, 5.5]
    y = [4.9, 5.8, 4.5, 6.6, 5.1]   # x consistently > y
    res = wsr.wilcoxon(x=x, y=y, alternative="greater")
    check(f"paired x>y: greater p small ({res['p_value']:.3f})", res["p_value"] < 0.1)

    # --- agrees with the sign-flip permutation test ---
    import permutation_test as pt
    rng = _R(5)
    diffs = [0.6 + 0.5 * rng.normal() for _ in range(25)]
    p_wsr = wsr.wilcoxon(diffs)["p_value"]
    p_perm = pt.sign_flip_test(diffs, n_resamples=4000, seed=1)["p_value"]
    check(f"agrees with sign-flip test (both significant: {p_wsr:.3f}, {p_perm:.3f})",
          (p_wsr < 0.05) == (p_perm < 0.05))

    # --- one-sided consistency ---
    diffs = [1.0 + 0.2 * _R(2).normal() for _ in range(15)]
    p_greater = wsr.wilcoxon(diffs, alternative="greater")["p_value"]
    p_less = wsr.wilcoxon(diffs, alternative="less")["p_value"]
    check("positive shift: greater<less", p_greater < p_less)

    # --- deterministic ---
    check("deterministic", wsr.wilcoxon(diffs) == wsr.wilcoxon(diffs))

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
