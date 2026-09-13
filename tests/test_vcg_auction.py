"""Tests for vcg_auction: second-price rule, efficiency, strategy-proofness, individual rationality."""

import os
import sys
from itertools import permutations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vcg_auction import (second_price_auction, vcg_allocate, additive_value,  # noqa: E402
                         bidder_utility)


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

    def nxt(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s

    def randint(self, lo, hi):
        return lo + (self.nxt() >> 8) % (hi - lo + 1)


def brute_efficient_total(n, m, valuations):
    """Max total value assigning each of m items to a bidder or nobody, additive values."""
    best = 0

    def rec(k, got):
        nonlocal best
        if k == m:
            best = max(best, got)
            return
        rec(k + 1, got)                       # item k unassigned
        for b in range(n):
            rec(k + 1, got + valuations[b][k])
    rec(0, 0)
    return best


def main():
    # ---- 1. second-price rule ---------------------------------------------------------
    check("second-price winner and price", second_price_auction([5, 9, 3, 7]) == (1, 7))
    check("second-price single bidder pays 0", second_price_auction([4]) == (0, 0))
    check("second-price ties to lowest index", second_price_auction([5, 5, 3]) == (0, 5))
    check("empty bids", second_price_auction([]) == (None, 0))
    # winner's surplus is first minus second
    w, p = second_price_auction([10, 6, 8])
    check("winner surplus = 1st - 2nd", 10 - p == 2)

    # ---- 2. VCG allocation is efficient (matches brute force) -------------------------
    rng = LCG(2024)
    eff_bad = 0
    for _ in range(200):
        n = rng.randint(2, 4)
        m = rng.randint(1, 3)
        vals = [[rng.randint(0, 12) for _ in range(m)] for _ in range(n)]
        r = vcg_allocate(n, list(range(m)), additive_value(vals))
        if abs(r["total_value"] - brute_efficient_total(n, m, vals)) > 1e-9:
            eff_bad += 1
    check("VCG allocation maximises total value (200 profiles)", eff_bad == 0, f"{eff_bad}")

    # ---- 3. strategy-proofness: truthful bidding is a best response -------------------
    rng = LCG(7)
    sp_bad = 0
    for _ in range(150):
        n = rng.randint(2, 3)
        m = rng.randint(1, 3)
        truevals = [[rng.randint(0, 10) for _ in range(m)] for _ in range(n)]
        truthful = vcg_allocate(n, list(range(m)), additive_value(truevals))
        for liar in range(n):
            u_true = truthful["utilities"][liar]
            for _ in range(12):
                lievals = [row[:] for row in truevals]
                lievals[liar] = [rng.randint(0, 15) for _ in range(m)]
                res = vcg_allocate(n, list(range(m)), additive_value(lievals))
                u_lie = bidder_utility(
                    res, liar, lambda b, tv=truevals[liar]: sum(tv[it] for it in b))
                if u_lie > u_true + 1e-9:
                    sp_bad += 1
    check("truthful bidding is a dominant strategy (no profitable lie)", sp_bad == 0, f"{sp_bad}")

    # ---- 4. individual rationality: utility always non-negative -----------------------
    ir_bad = 0
    pay_bad = 0
    for _ in range(200):
        n = rng.randint(2, 4)
        m = rng.randint(1, 3)
        vals = [[rng.randint(0, 12) for _ in range(m)] for _ in range(n)]
        r = vcg_allocate(n, list(range(m)), additive_value(vals))
        for i in range(n):
            if r["utilities"][i] < -1e-9:
                ir_bad += 1
            # payment never exceeds the value the winner gets
            won_value = sum(vals[i][it] for it in r["allocation"][i])
            if r["payments"][i] > won_value + 1e-9:
                pay_bad += 1
            if r["payments"][i] < -1e-9:
                pay_bad += 1
    check("individually rational (utility >= 0)", ir_bad == 0, f"{ir_bad}")
    check("payments in [0, winner's value]", pay_bad == 0, f"{pay_bad}")

    # ---- 5. hand-checked instance -----------------------------------------------------
    # 2 items, 3 bidders; efficient assignment gives item0 to b0, item1 to b1
    vals = [[10, 4], [6, 8], [3, 3]]
    r = vcg_allocate(3, [0, 1], additive_value(vals))
    check("efficient allocation splits the items", r["allocation"][0] == [0]
          and r["allocation"][1] == [1] and r["allocation"][2] == [])
    check("total value is 18", r["total_value"] == 18)
    # b0's payment: others' best without b0 is b1 takes both (8+6=14) or b1:8,b2:3 -> 11; with b0 present
    #   others get 8 (b1's item1). Externality = 14 - 8 = 6.
    check("VCG payment is the externality", r["payments"][0] == 6, f"{r['payments'][0]}")

    # ---- 6. single-item VCG reduces to second price -----------------------------------
    vals = [[7], [3], [5]]
    r = vcg_allocate(3, [0], additive_value(vals))
    check("single-item VCG: highest bidder wins", r["allocation"][0] == [0])
    check("single-item VCG price = second highest", r["payments"][0] == 5, f"{r['payments'][0]}")

    # ---- 7. everyone wins something when items >= bidders and values positive ---------
    vals = [[5, 1], [1, 5]]
    r = vcg_allocate(2, [0, 1], additive_value(vals))
    check("complementary values -> each gets their favourite",
          r["allocation"][0] == [0] and r["allocation"][1] == [1])

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
