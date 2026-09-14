"""Validate Mann-Kendall: increasing/decreasing/no-trend, monotone invariance, ties, Sen slope, nonlinear."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import mann_kendall_trend as mk


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
    print("Mann-Kendall trend tests")

    # --- clean increasing series ---
    rng = _R(1)
    x = [0.5 * i + rng.normal() for i in range(40)]
    res = mk.mann_kendall(x)
    check(f"increasing: positive S ({res['S']})", res["S"] > 0)
    check(f"increasing: tiny p ({res['p_value']:.2e})", res["p_value"] < 0.001)
    check("increasing: trend label", res["trend"] == "increasing")
    check(f"increasing: Sen slope near 0.5 ({res['sen_slope']:.3f})", abs(res["sen_slope"] - 0.5) < 0.15)

    # --- clean decreasing series ---
    x = [-0.4 * i + rng.normal() for i in range(40)]
    res = mk.mann_kendall(x)
    check(f"decreasing: negative S ({res['S']})", res["S"] < 0)
    check("decreasing: trend label", res["trend"] == "decreasing")
    check("decreasing: negative Sen slope", res["sen_slope"] < 0)

    # --- no trend: pure noise ---
    rng = _R(7)
    x = [rng.normal() for _ in range(50)]
    res = mk.mann_kendall(x)
    check(f"no trend: large p ({res['p_value']:.3f})", res["p_value"] > 0.1)
    check("no trend: label", res["trend"] == "no trend")

    # --- monotone-transform invariance of S (ranks only) ---
    x = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0]
    s1 = mk.mann_kendall(x)["S"]
    x2 = [math.exp(v) for v in x]
    s2 = mk.mann_kendall(x2)["S"]
    check("S invariant under monotone transform", s1 == s2)

    # --- nonlinear but monotone trend still detected ---
    x = [math.log(i + 1) + 0.05 * _R(3).normal() for i in range(40)]
    # use a proper noisy log trend
    rng = _R(3)
    x = [math.log(i + 1) + 0.1 * rng.normal() for i in range(40)]
    res = mk.mann_kendall(x)
    check(f"nonlinear monotone trend detected ({res['trend']})", res["trend"] == "increasing")

    # --- tie handling: repeated values don't invent a trend ---
    x = [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2]  # oscillating, no net trend
    res = mk.mann_kendall(x)
    check(f"ties: no spurious trend ({res['trend']})", res["trend"] == "no trend")
    # variance correction reduces var below the no-tie formula
    n = len(x)
    var_notie = n * (n - 1) * (2 * n + 5) / 18.0
    check("tie correction lowers variance", res["var_s"] < var_notie)

    # --- tau matches S / (n(n-1)/2) ---
    x = [rng.normal() + 0.3 * i for i in range(30)]
    res = mk.mann_kendall(x)
    n = len(x)
    check("tau == S/(n(n-1)/2)", abs(res["tau"] - res["S"] / (n * (n - 1) / 2)) < 1e-12)

    # --- Sen slope matches Theil-Sen directly ---
    import theil_sen as ts
    slope_ts, _ = ts.theil_sen(list(range(len(x))), x)
    check("Sen slope == Theil-Sen slope", abs(mk.sen_slope(x) - slope_ts) < 1e-12)

    # --- one-sided tests ---
    x = [0.5 * i + _R(2).normal() for i in range(30)]
    p_inc = mk.mann_kendall(x, alternative="increasing")["p_value"]
    p_dec = mk.mann_kendall(x, alternative="decreasing")["p_value"]
    check(f"increasing one-sided p small ({p_inc:.2e})", p_inc < 0.01)
    check(f"decreasing one-sided p large ({p_dec:.3f})", p_dec > 0.95)

    # --- perfect monotone: S = n(n-1)/2 ---
    x = list(range(10))
    res = mk.mann_kendall(x)
    check("perfect increasing S = n(n-1)/2", res["S"] == 10 * 9 // 2)
    check("perfect increasing tau == 1", abs(res["tau"] - 1.0) < 1e-12)

    # --- deterministic ---
    check("deterministic", mk.mann_kendall(x) == mk.mann_kendall(x))

    # --- rejects too-short series ---
    try:
        mk.mann_kendall([1, 2])
        check("rejects < 3 points", False)
    except ValueError:
        check("rejects < 3 points", True)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
