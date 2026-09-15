"""Validate Simpson's paradox: detect reversal on kidney-stone & Berkeley data, confirm adjustment fixes it."""

import itertools
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import simpsons_paradox as sp


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


# Kidney-stone study (Charig 1986). Group A = open surgery, B = percutaneous nephrolithotomy.
# (a_succ, a_tot, b_succ, b_tot) per stratum. Small stones then large stones.
KIDNEY = [
    (81, 87, 234, 270),    # small stones: A 93.1%, B 86.7%  -> A wins
    (192, 263, 55, 80),    # large stones: A 73.0%, B 68.8%  -> A wins
]
# Berkeley 1973 admissions, simplified 2-dept confounder. A = men, B = women.
# (men_admit, men_apply, women_admit, women_apply)
BERKELEY = [
    (512, 825, 89, 108),   # dept A (easy): men 62.1%, women 82.4% -> women win
    (16, 373, 245, 1100),  # dept F (hard): men 4.3%,  women 22.3% -> women win
]


def main():
    print("Simpson's paradox tests")

    # --- kidney stones: A wins both strata but loses the pooled rate ---
    pr_a, pr_b = sp.pooled_rates(KIDNEY)
    check(f"kidney pooled: A {pr_a:.3f} < B {pr_b:.3f} (A looks worse)", pr_a < pr_b)
    srates = sp.stratum_rates(KIDNEY)
    check("kidney: A wins small stones", srates[0][0] > srates[0][1])
    check("kidney: A wins large stones", srates[1][0] > srates[1][1])
    check("kidney reversal detected", sp.is_reversal(KIDNEY))

    # --- pooled OR points the wrong way (<1, favouring B); MH points right (>1, favouring A) ---
    p_or = sp.pooled_odds_ratio(KIDNEY)
    mh = sp.mantel_haenszel_or(KIDNEY)
    check(f"kidney pooled OR {p_or:.3f} < 1 (wrongly favours B)", p_or < 1)
    check(f"kidney MH OR {mh:.3f} > 1 (correctly favours A)", mh > 1)

    # --- standardization also recovers A's advantage ---
    sa, sb = sp.standardized_rates(KIDNEY)
    check(f"kidney standardized: A {sa:.3f} > B {sb:.3f}", sa > sb)

    # --- confounder imbalance: A got far more of the hard (large-stone) cases ---
    alloc = sp.allocation(KIDNEY)
    a_large = alloc[1][0]
    b_large = alloc[1][1]
    check(f"kidney: A's large-stone share {a_large:.2f} >> B's {b_large:.2f}", a_large > b_large)

    # --- Berkeley: women win both depts but lose pooled ---
    bpr_a, bpr_b = sp.pooled_rates(BERKELEY)  # A=men, B=women
    check(f"berkeley pooled: men {bpr_a:.3f} > women {bpr_b:.3f} (men look favoured)", bpr_a > bpr_b)
    bsr = sp.stratum_rates(BERKELEY)
    check("berkeley: women win dept A", bsr[0][1] > bsr[0][0])
    check("berkeley: women win dept F", bsr[1][1] > bsr[1][0])
    check("berkeley reversal detected", sp.is_reversal(BERKELEY))
    bmh = sp.mantel_haenszel_or(BERKELEY)  # OR for men vs women
    check(f"berkeley MH OR {bmh:.3f} < 1 (correctly favours women)", bmh < 1)

    # --- no paradox when there is no confounding: equal case-mix, A better everywhere ---
    clean = [(90, 100, 80, 100), (45, 50, 40, 50)]  # A wins both, balanced sizes
    check("clean data: not a reversal", not sp.is_reversal(clean))
    cpr_a, cpr_b = sp.pooled_rates(clean)
    check("clean: pooled agrees (A wins)", cpr_a > cpr_b)

    # --- a tie in a stratum is not a clean reversal ---
    tied = [(50, 100, 50, 100), (81, 87, 234, 270)]
    check("stratum tie -> not a reversal", not sp.is_reversal(tied))

    # --- single-stratum odds ratio matches hand calc ---
    # small stones: (81*36)/(6*234) = 2916/1404 = 2.0769...
    or0 = sp.odds_ratio(81, 87, 234, 270)
    check(f"single-stratum OR exact ({or0:.4f})", abs(or0 - (81 * 36) / (6 * 234)) < 1e-9)

    # --- MH OR equals the single OR when there is only one stratum ---
    one = [KIDNEY[0]]
    check("MH == single OR for one stratum", abs(sp.mantel_haenszel_or(one) - or0) < 1e-9)

    # --- brute-force: is_reversal <=> pooled sign opposite to unanimous non-zero stratum sign ---
    def naive_reversal(strata):
        pr_a, pr_b = sp.pooled_rates(strata)
        ps = sp._sign(pr_a - pr_b)
        if ps == 0:
            return False
        signs = [sp._sign(ra - rb) for ra, rb in sp.stratum_rates(strata)]
        if any(s == 0 for s in signs) or len(set(signs)) != 1:
            return False
        return signs[0] != ps

    mismatches = 0
    checked = 0
    # sweep small integer 2x2 tables over 2 strata
    rng_vals = range(0, 6)
    combos = itertools.product(rng_vals, repeat=8)
    for i, (a0, af0, b0, bf0, a1, af1, b1, bf1) in enumerate(combos):
        if i % 37 != 0:  # subsample the ~1.6M space for speed
            continue
        at0, bt0, at1, bt1 = a0 + af0, b0 + bf0, a1 + af1, b1 + bf1
        if min(at0, bt0, at1, bt1) == 0:
            continue
        strata = [(a0, at0, b0, bt0), (a1, at1, b1, bt1)]
        checked += 1
        if sp.is_reversal(strata) != naive_reversal(strata):
            mismatches += 1
    check(f"brute-force reversal definition ({checked} tables, {mismatches} mismatches)", mismatches == 0)

    # --- reversal is symmetric under swapping the two groups ---
    swapped = [(b, bt, a, at) for (a, at, b, bt) in KIDNEY]
    check("reversal symmetric under group swap", sp.is_reversal(swapped) == sp.is_reversal(KIDNEY))

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
