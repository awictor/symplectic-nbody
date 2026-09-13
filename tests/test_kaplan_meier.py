"""Tests for Kaplan-Meier: no-censoring == 1-ECDF, monotone, textbook example, Greenwood, log-rank."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kaplan_meier import (  # noqa: E402
    kaplan_meier,
    survival_at,
    median_survival,
    empirical_survival,
    logrank_test,
    confidence_band,
)


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


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    # ---- 1. with no censoring, KM == 1 - empirical CDF ----------------------------------
    times = [2, 3, 3, 5, 7, 8, 8, 8, 10]
    events = [1] * len(times)
    et, S, var = kaplan_meier(times, events)
    esurv = empirical_survival(times)
    # compare S(t) at each distinct time to the empirical survival
    ok = True
    for t, es in esurv:
        if abs(survival_at(et, S, t) - es) > 1e-9:
            ok = False
            break
    check("no censoring: KM == 1 - ECDF", ok)

    # ---- 2. survival is monotone non-increasing, starts <= 1 ----------------------------
    check("survival non-increasing", all(S[i + 1] <= S[i] + 1e-12 for i in range(len(S) - 1)))
    check("survival <= 1", all(s <= 1.0 + 1e-12 for s in S))
    check("survival(before first event) = 1", abs(survival_at(et, S, min(times) - 1) - 1.0) < 1e-12)

    # ---- 3. textbook product-limit example ----------------------------------------------
    # 6 subjects, times [6,6,6,7,10,13], first three at t=6 are events, rest censored+events mix.
    # Classic: times 6(event x3), 7(event), 10(censored), 13(event)
    t3 = [6, 6, 6, 7, 10, 13]
    e3 = [1, 1, 1, 1, 0, 1]
    et3, S3, v3 = kaplan_meier(t3, e3)
    # at t=6: n=6, d=3 -> S=1 - 3/6 = 0.5
    # at t=7: n=3, d=1 -> S=0.5 * (1 - 1/3) = 0.3333
    # t=10 censored (no drop). at t=13: n=1, d=1 -> S=0.3333*(1-1/1)=0
    check("textbook S(6) = 0.5", abs(survival_at(et3, S3, 6) - 0.5) < 1e-9, f"{survival_at(et3, S3, 6)}")
    check("textbook S(7) = 1/3", abs(survival_at(et3, S3, 7) - 1 / 3) < 1e-9, f"{survival_at(et3, S3, 7)}")
    check("textbook S(11) still 1/3 (censoring no drop)",
          abs(survival_at(et3, S3, 11) - 1 / 3) < 1e-9, f"{survival_at(et3, S3, 11)}")
    check("textbook S(13) = 0", abs(survival_at(et3, S3, 13) - 0.0) < 1e-9, f"{survival_at(et3, S3, 13)}")

    # ---- 4. each drop matches the product-limit factor ----------------------------------
    # verify the multiplicative structure directly
    check("product-limit structure", abs(S3[1] - S3[0] * (1 - 1 / 3)) < 1e-9)

    # ---- 5. Greenwood variance matches a direct recomputation ---------------------------
    # recompute Greenwood by hand for the textbook example
    # after t=6: S=0.5, gw = 3/(6*3)=1/6, var = 0.25 * 1/6
    check("Greenwood var at t=6", abs(v3[0] - 0.25 * (3 / (6 * 3))) < 1e-9, f"{v3[0]}")

    # ---- 6. median survival ------------------------------------------------------------
    # for t3, S drops to 0.5 exactly at t=6
    check("median survival = 6", median_survival(et3, S3) == 6, f"{median_survival(et3, S3)}")
    # a curve that never reaches 0.5
    t_hi = [1, 2, 3]
    e_hi = [1, 0, 0]  # only one event out of three
    et_hi, S_hi, _ = kaplan_meier(t_hi, e_hi)
    check("median None when S never hits 0.5", median_survival(et_hi, S_hi) is None,
          f"{median_survival(et_hi, S_hi)}")

    # ---- 7. confidence band brackets the survival curve ---------------------------------
    band = confidence_band(et3, S3, v3)
    check("confidence band brackets S", all(band[i][0] <= S3[i] <= band[i][1] for i in range(len(S3))))
    check("band clipped to [0,1]", all(0 <= lo <= 1 and 0 <= hi <= 1 for lo, hi in band))

    # ---- 8. log-rank ~ 0 for identical groups -------------------------------------------
    rng = _lcg(1)
    t1 = [1 + int(rng() * 20) for _ in range(50)]
    e1 = [1] * 50
    t2 = [1 + int(rng() * 20) for _ in range(50)]
    e2 = [1] * 50
    chi2, O1, E1 = logrank_test(t1, e1, t2, e2)
    check("log-rank small for similar groups", chi2 < 6.0, f"chi2={chi2:.3f}")

    # ---- 9. log-rank large for well-separated groups ------------------------------------
    # group A dies early, group B dies late
    tA = [1, 2, 2, 3, 3, 4, 4, 5]
    eA = [1] * 8
    tB = [15, 16, 16, 17, 18, 19, 20, 21]
    eB = [1] * 8
    chi2, O1, E1 = logrank_test(tA, eA, tB, eB)
    check("log-rank large for separated groups", chi2 > 10.0, f"chi2={chi2:.3f}")

    # ---- 10. observed vs expected balance ----------------------------------------------
    # total observed events across groups should equal total events
    total_events = sum(eA) + sum(eB)
    # O1 + O2 = total; E1 + E2 = total too
    check("log-rank O1 <= total events", 0 <= O1 <= total_events, f"O1={O1}")

    # ---- 11. all-censored -> flat survival at 1 -----------------------------------------
    et_c, S_c, _ = kaplan_meier([5, 6, 7], [0, 0, 0])
    check("all censored -> no drops", et_c == [] and S_c == [])
    check("all censored S(anything) = 1", abs(survival_at(et_c, S_c, 100) - 1.0) < 1e-12)

    # ---- 12. single subject -------------------------------------------------------------
    et1, S1, _ = kaplan_meier([4], [1])
    check("single event S(4) = 0", abs(survival_at(et1, S1, 4) - 0.0) < 1e-12)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
