"""Validate Friedman: null vs treatment effect, block-effect removal, ties, Kendall W, chi2 df."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import friedman_test as ft


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
    print("Friedman test tests")

    # --- no treatment effect: random within blocks -> large p ---
    rng = _R(2)
    data = [[rng.normal() for _ in range(4)] for _ in range(20)]
    res = ft.friedman(data)
    check(f"null: large p ({res['p_value']:.3f})", res["p_value"] > 0.1)
    check("df == k-1", res["df"] == 3)

    # --- clear treatment effect: treatment j gets +j each block ---
    rng = _R(3)
    data = []
    for _ in range(20):
        base = rng.normal()
        data.append([base + 0 + 0.3 * rng.normal(),
                     base + 2 + 0.3 * rng.normal(),
                     base + 4 + 0.3 * rng.normal(),
                     base + 6 + 0.3 * rng.normal()])
    res = ft.friedman(data)
    check(f"treatment effect: tiny p ({res['p_value']:.2e})", res["p_value"] < 1e-6)
    check("treatment effect: large Q", res["Q"] > 20)
    # avg ranks should increase with treatment index
    check("avg ranks ordered by treatment", res["avg_ranks"] == sorted(res["avg_ranks"]))

    # --- block-effect removal: adding a constant to a whole block changes nothing ---
    data2 = [[data[i][j] + 100 * i for j in range(4)] for i in range(len(data))]
    res2 = ft.friedman(data2)
    check("additive block effect removed (same Q)", abs(res["Q"] - res2["Q"]) < 1e-9)

    # --- Kendall's W in [0,1], near 1 for strong agreement ---
    check("Kendall W in [0,1]", 0 <= res["kendall_w"] <= 1)
    check(f"strong agreement -> high W ({res['kendall_w']:.2f})", res["kendall_w"] > 0.7)

    # --- perfect agreement: every block ranks treatments identically -> W = 1 ---
    perfect = [[1.0, 2.0, 3.0, 4.0] for _ in range(10)]
    res_p = ft.friedman(perfect)
    check(f"perfect agreement W == 1 ({res_p['kendall_w']:.3f})", abs(res_p["kendall_w"] - 1.0) < 1e-9)

    # --- k=2 reduces to a sign-test-style comparison ---
    # treatment B beats A in most blocks
    rng = _R(5)
    data_k2 = [[rng.normal(), rng.normal() + 1.5] for _ in range(25)]
    res_k2 = ft.friedman(data_k2)
    check(f"k=2 detects difference ({res_k2['p_value']:.3f})", res_k2["p_value"] < 0.05)
    check("k=2: df == 1", res_k2["df"] == 1)

    # --- ranking is WITHIN blocks (distinguishes from Kruskal-Wallis) ---
    # a block whose values are all high still contributes ranks 1..k, not high ranks
    ranks = ft._rank_within_blocks([[10, 20, 30], [1, 2, 3]])
    check("within-block ranks identical for parallel blocks", ranks[0] == ranks[1] == [1.0, 2.0, 3.0])

    # --- tie handling within a block: average ranks ---
    ranks = ft._rank_within_blocks([[5, 5, 8]])
    check("tied within-block average ranks", ranks[0] == [1.5, 1.5, 3.0])

    # --- rank sums total to n * k(k+1)/2 ---
    n, k = len(data), 4
    total = sum(res["rank_sums"])
    check("rank sums total correct", abs(total - n * k * (k + 1) / 2) < 1e-9)

    # --- deterministic ---
    check("deterministic", ft.friedman(data) == ft.friedman(data))

    # --- rejects too few blocks/treatments ---
    try:
        ft.friedman([[1, 2, 3]])
        check("rejects single block", False)
    except ValueError:
        check("rejects single block", True)
    try:
        ft.friedman([[1], [2]])
        check("rejects single treatment", False)
    except ValueError:
        check("rejects single treatment", True)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
