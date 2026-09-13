"""Tests for rt_scheduling: EDF test exact vs sim, RM sufficient, EDF dominates RM, hyperperiod."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rt_scheduling import (Task, utilisation, liu_layland_bound, rm_schedulable_ll,  # noqa: E402
                           edf_schedulable, simulate, hyperperiod, rm_schedulable_exact,
                           edf_schedulable_exact)


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


def main():
    # ---- 1. utilisation and Liu-Layland bound -----------------------------------------
    t = [Task(1, 4), Task(2, 8)]
    check("utilisation computed", abs(utilisation(t) - 0.5) < 1e-12)
    check("LL bound for 1 task is 1.0", abs(liu_layland_bound(1) - 1.0) < 1e-12)
    check("LL bound for 2 tasks ~ 0.828", abs(liu_layland_bound(2) - 0.8284) < 1e-3)
    check("LL bound decreases toward ln2", liu_layland_bound(100) < 0.71 and liu_layland_bound(100) > 0.69)

    # ---- 2. EDF exact test (U <= 1) matches simulation --------------------------------
    rng = LCG(2024)
    edf_bad = 0
    tested = 0
    for _ in range(400):
        n = rng.randint(2, 4)
        tasks = [Task(rng.randint(1, 4), rng.randint(2, 9)) for _ in range(n)]
        if hyperperiod(tasks) > 5000:
            continue
        tested += 1
        if edf_schedulable(tasks) != edf_schedulable_exact(tasks):
            edf_bad += 1
    check(f"EDF test (U<=1) is exact vs simulation ({tested} sets)", edf_bad == 0, f"{edf_bad}")

    # ---- 3. RM Liu-Layland test is SUFFICIENT (never over-optimistic) -----------------
    rm_bad = 0
    tested = 0
    for _ in range(400):
        n = rng.randint(2, 4)
        tasks = [Task(rng.randint(1, 3), rng.randint(3, 9)) for _ in range(n)]
        if hyperperiod(tasks) > 5000:
            continue
        tested += 1
        if rm_schedulable_ll(tasks) and not rm_schedulable_exact(tasks):
            rm_bad += 1
    check(f"RM LL test is sufficient ({tested} sets)", rm_bad == 0, f"{rm_bad}")

    # ---- 4. RM test is NOT necessary (schedulable sets above the bound exist) ---------
    # 2 tasks with U between 0.828 and 1 that are still RM-schedulable
    over_bound_schedulable = 0
    for _ in range(200):
        tasks = [Task(rng.randint(1, 4), rng.randint(2, 8)) for _ in range(2)]
        if hyperperiod(tasks) > 5000:
            continue
        u = utilisation(tasks)
        if u > liu_layland_bound(2) and u <= 1.0 and rm_schedulable_exact(tasks):
            over_bound_schedulable += 1
    check("RM test is not necessary (above-bound sets still schedulable)",
          over_bound_schedulable > 0, "found none")

    # ---- 5. EDF dominates RM: anything RM schedules, EDF schedules too ----------------
    edf_dominates = True
    for _ in range(400):
        n = rng.randint(2, 4)
        tasks = [Task(rng.randint(1, 3), rng.randint(2, 8)) for _ in range(n)]
        if hyperperiod(tasks) > 5000:
            continue
        if rm_schedulable_exact(tasks) and not edf_schedulable_exact(tasks):
            edf_dominates = False
    check("EDF dominates RM (schedules everything RM does)", edf_dominates)

    # ---- 6. classic Liu-Layland example ----------------------------------------------
    # tasks (C,T): (1,2),(1,3) -> U = 0.5+0.333 = 0.833; LL(2)=0.828, so LL says maybe not,
    # but EDF (U<=1) yes and simulation should schedule it
    tasks = [Task(1, 2), Task(1, 3)]
    check("EDF schedules U=0.833 set", edf_schedulable(tasks) and edf_schedulable_exact(tasks))
    check("this set is RM-schedulable by simulation", rm_schedulable_exact(tasks))

    # ---- 7. overload: U slightly above 1 misses under both ----------------------------
    tasks = [Task(2, 3), Task(2, 3)]     # U = 4/3 > 1
    check("overloaded set fails EDF test", not edf_schedulable(tasks))
    check("overloaded set misses in EDF simulation", not edf_schedulable_exact(tasks))
    check("overloaded set misses in RM simulation", not rm_schedulable_exact(tasks))

    # ---- 8. hyperperiod is the LCM of periods -----------------------------------------
    check("hyperperiod = lcm", hyperperiod([Task(1, 4), Task(1, 6), Task(1, 8)]) == 24)
    check("coprime periods -> product", hyperperiod([Task(1, 3), Task(1, 5)]) == 15)

    # ---- 9. a fully-utilised harmonic set is schedulable by both ----------------------
    # harmonic periods, U = 1 exactly: (1,2),(1,4),(2,8) -> 0.5+0.25+0.25 = 1.0
    tasks = [Task(1, 2), Task(1, 4), Task(2, 8)]
    check("U=1 harmonic set schedulable by EDF", edf_schedulable_exact(tasks))
    check("U=1 harmonic set schedulable by RM", rm_schedulable_exact(tasks))

    # ---- 10. simulation reports the missing task -------------------------------------
    ok, misses = simulate([Task(2, 3), Task(2, 3)], "edf")
    check("simulation reports deadline misses", not ok and len(misses) > 0)

    # ---- 11. single task always schedulable if C <= T --------------------------------
    check("single task C<=T schedulable", edf_schedulable_exact([Task(3, 5)])
          and rm_schedulable_exact([Task(3, 5)]))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
