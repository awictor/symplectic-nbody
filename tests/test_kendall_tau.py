"""Validate Kendall tau: perfect monotone, independence, tie handling, invariance, brute-force match."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import kendall_tau as kt


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)


def brute_tau_a(x, y):
    n = len(x)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (x[i] - x[j]) * (y[i] - y[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    return (c - d) / (n * (n - 1) / 2)


def main():
    print("Kendall tau tests")

    # --- perfectly increasing -> tau = 1 ---
    x = list(range(10))
    y = [2 * v + 1 for v in x]
    check("perfect increasing tau_b == 1", abs(kt.tau_b(x, y) - 1.0) < 1e-12)
    check("perfect increasing tau_a == 1", abs(kt.tau_a(x, y) - 1.0) < 1e-12)

    # --- perfectly decreasing -> tau = -1 ---
    y = [-3 * v for v in x]
    check("perfect decreasing tau_b == -1", abs(kt.tau_b(x, y) + 1.0) < 1e-12)

    # --- monotonic transform invariance ---
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y = [1.5, 0.5, 3.0, 2.5, 5.0, 4.0]
    t1 = kt.tau_b(x, y)
    fx = [math.exp(v) for v in x]
    fy = [v ** 3 for v in y]  # monotone increasing cube preserves order for positive y... use exp for safety
    fy = [math.exp(v) for v in y]
    t2 = kt.tau_b(fx, fy)
    check(f"monotone-transform invariant ({t1:.4f} vs {t2:.4f})", abs(t1 - t2) < 1e-12)

    # --- independence -> tau near 0, large p ---
    rng = _R(3)
    x = [rng.u() for _ in range(60)]
    y = [rng.u() for _ in range(60)]
    res = kt.kendall(x, y)
    check(f"independent tau near 0 ({res['tau_b']:.3f})", abs(res["tau_b"]) < 0.25)
    check(f"independent large p ({res['p_value']:.3f})", res["p_value"] > 0.1)

    # --- strong association -> small p ---
    x = list(range(30))
    rng = _R(5)
    y = [v + 4 * rng.u() for v in x]  # increasing + noise
    res = kt.kendall(x, y)
    check(f"strong association small p ({res['p_value']:.2e})", res["p_value"] < 0.001)
    check("strong association positive tau", res["tau_b"] > 0.5)

    # --- tau matches brute-force pair enumeration (no ties) ---
    rng = _R(9)
    x = [rng.u() for _ in range(25)]
    y = [rng.u() for _ in range(25)]
    check(f"tau_a matches brute force", abs(kt.tau_a(x, y) - brute_tau_a(x, y)) < 1e-12)

    # --- concordant + discordant == untied pairs (no ties) ---
    c, d, xt, yt = kt._counts(x, y)
    n0 = 25 * 24 // 2
    check("C + D == all pairs (no ties)", c + d == n0 and xt == 0 and yt == 0)

    # --- tie handling: tau-b reaches 1 on perfect monotone-with-ties ---
    x = [1, 1, 2, 2, 3, 3]
    y = [1, 1, 2, 2, 3, 3]  # identical -> perfect association with ties
    check(f"tau_b == 1 with ties ({kt.tau_b(x, y):.4f})", abs(kt.tau_b(x, y) - 1.0) < 1e-9)
    # tau-a underestimates (< 1) because it doesn't correct for ties
    check("tau_a < 1 with ties (no correction)", kt.tau_a(x, y) < 1.0)

    # --- ties counted correctly ---
    x = [1, 1, 2, 3]
    y = [5, 6, 7, 8]
    c, d, xt, yt = kt._counts(x, y)
    # only tie is x[0]==x[1]; that pair is x-tied
    check("x-tie counted", xt == 1 and yt == 0)

    # --- one-sided tests ---
    x = list(range(20))
    y = [v + 2 * _R(1).u() for v in x]   # increasing
    p_g = kt.kendall(x, y, alternative="greater")["p_value"]
    p_l = kt.kendall(x, y, alternative="less")["p_value"]
    check(f"increasing: 'greater' p small ({p_g:.2e})", p_g < 0.01)
    check(f"increasing: 'less' p large ({p_l:.3f})", p_l > 0.95)

    # --- symmetry: tau(x,y) == tau(y,x) ---
    rng = _R(11)
    x = [rng.u() for _ in range(20)]
    y = [rng.u() for _ in range(20)]
    check("tau symmetric in arguments", abs(kt.tau_b(x, y) - kt.tau_b(y, x)) < 1e-12)

    # --- deterministic ---
    check("deterministic", kt.kendall(x, y) == kt.kendall(x, y))

    # --- length mismatch raises ---
    try:
        kt.kendall([1, 2, 3], [1, 2])
        check("rejects length mismatch", False)
    except ValueError:
        check("rejects length mismatch", True)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
