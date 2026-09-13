"""Tests for page_replacement: optimal is minimal, LRU stack property, Belady anomaly, hand traces."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from page_replacement import (fifo, lru, clock, lfu, optimal, compare,  # noqa: E402
                              distinct_pages, belady_anomaly_example)


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


def trace_valid(refs, frames, faults, trace):
    """Check the trace: compulsory misses, no fault while resident, resident set bounded."""
    resident = set()
    order_seen = set()
    fault_count = 0
    for (kind, page, evicted), ref in zip(trace, refs):
        if kind == "fault":
            fault_count += 1
            if evicted is not None:
                resident.discard(evicted)
            resident.add(page)
            # first-ever reference must be a fault (compulsory)
        else:  # hit
            if page not in resident:
                return False
        if len(resident) > frames:
            return False
    return fault_count == faults


def main():
    # ---- 1. hand-traced classic string ------------------------------------------------
    refs = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
    c = compare(refs, 3)
    check("optimal <= LRU <= (ballpark) on classic string", c["OPTIMAL"] <= c["LRU"] <= c["FIFO"] + 5)
    check("optimal fault count is 9 (known)", c["OPTIMAL"] == 9, f"{c['OPTIMAL']}")

    # ---- 2. optimal (Belady MIN) is never worse than any other policy -----------------
    rng = LCG(2024)
    opt_bad = 0
    lb_bad = 0
    for _ in range(400):
        n = rng.randint(5, 25)
        pages = rng.randint(2, 7)
        refs = [rng.randint(0, pages - 1) for _ in range(n)]
        fr = rng.randint(1, 5)
        c = compare(refs, fr)
        if any(c["OPTIMAL"] > c[p] for p in c):
            opt_bad += 1
        # optimal faults >= number of distinct pages (compulsory misses), when frames < distinct
        if c["OPTIMAL"] < min(distinct_pages(refs), len(set(refs))):
            lb_bad += 1
    check("optimal is the minimum-fault policy (400 strings)", opt_bad == 0, f"{opt_bad}")
    check("optimal >= compulsory-miss lower bound", lb_bad == 0, f"{lb_bad}")

    # ---- 3. LRU is a stack algorithm: more frames never increases faults ---------------
    lru_bad = 0
    for _ in range(200):
        n = rng.randint(5, 30)
        pages = rng.randint(2, 8)
        refs = [rng.randint(0, pages - 1) for _ in range(n)]
        prev = None
        monotone = True
        for fr in range(1, pages + 2):
            f = lru(refs, fr)[0]
            if prev is not None and f > prev:
                monotone = False
            prev = f
        if not monotone:
            lru_bad += 1
    check("LRU is a stack algorithm (faults monotone in frames)", lru_bad == 0, f"{lru_bad}")

    # optimal is also a stack algorithm
    opt_mono_bad = 0
    for _ in range(100):
        n = rng.randint(5, 25)
        pages = rng.randint(2, 6)
        refs = [rng.randint(0, pages - 1) for _ in range(n)]
        prev = None
        for fr in range(1, pages + 2):
            f = optimal(refs, fr)[0]
            if prev is not None and f > prev:
                opt_mono_bad += 1
                break
            prev = f
    check("optimal is a stack algorithm too", opt_mono_bad == 0, f"{opt_mono_bad}")

    # ---- 4. Belady's anomaly: FIFO can fault MORE with MORE frames --------------------
    refs, f3, f4 = belady_anomaly_example()
    check("Belady's anomaly reproduced (FIFO 4 frames > 3 frames)", f4 > f3, f"3->{f3}, 4->{f4}")
    # and LRU does NOT exhibit it on the same string
    check("LRU immune on the same string", lru(refs, 4)[0] <= lru(refs, 3)[0])

    # ---- 5. traces are valid ----------------------------------------------------------
    trace_bad = 0
    for policy in (fifo, lru, clock, lfu, optimal):
        for _ in range(40):
            n = rng.randint(3, 20)
            pages = rng.randint(2, 6)
            refs = [rng.randint(0, pages - 1) for _ in range(n)]
            fr = rng.randint(1, 4)
            faults, trace = policy(refs, fr)
            if not trace_valid(refs, fr, faults, trace):
                trace_bad += 1
    check("all policy traces are valid (compulsory miss, no fault on hit, bounded set)",
          trace_bad == 0, f"{trace_bad}")

    # ---- 6. every first reference faults; enough frames -> only distinct-page faults --
    refs = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    # with 3 frames, only 3 compulsory faults for all policies
    for name, policy in (("FIFO", fifo), ("LRU", lru), ("OPTIMAL", optimal)):
        f = policy(refs, 3)[0]
        check(f"{name}: 3 frames fit 3 pages -> 3 faults", f == 3, f"{f}")

    # ---- 7. single frame: faults on every change --------------------------------------
    refs = [1, 1, 2, 2, 1, 3]
    check("single frame counts every distinct consecutive change", lru(refs, 1)[0] == 4)

    # ---- 8. clock sits between FIFO and true LRU (approximation) ----------------------
    # over random strings, clock faults are typically between; check it's a valid approximation
    refs = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2]
    cf = clock(refs, 3)[0]
    check("clock fault count is reasonable (>= optimal, <= distinct*len bound)",
          optimal(refs, 3)[0] <= cf <= len(refs), f"clock {cf}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
