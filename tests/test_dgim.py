"""Tests for dgim: estimate within error bound vs exact window, log memory, edge cases."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dgim import DGIM, ExactWindow  # noqa: E402


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

    def bit(self, p_num=1, p_den=2):
        return 1 if (self.nxt() >> 8) % p_den < p_num else 0


def main():
    # ---- 1. estimate within the guaranteed error band vs exact, several densities -----
    for (window, pnum, pden, seed) in [(100, 1, 2, 1), (256, 1, 4, 7), (500, 3, 4, 42),
                                       (64, 1, 8, 100)]:
        rng = LCG(seed)
        d = DGIM(window, buckets_per_size=2)
        e = ExactWindow(window)
        worst_rel = 0.0
        for _ in range(6000):
            b = rng.bit(pnum, pden)
            d.push(b)
            e.push(b)
            est = d.count()
            exact = e.count()
            if exact > 0:
                rel = abs(est - exact) / exact
                worst_rel = max(worst_rel, rel)
        # DGIM's guaranteed bound with r=2 is 50%
        check(f"within 50% error (window={window}, density={pnum}/{pden})", worst_rel <= 0.5 + 1e-9,
              f"worst rel err {worst_rel:.3f}")

    # ---- 2. more buckets per size -> tighter empirical error --------------------------
    rng = LCG(2024)
    stream = [rng.bit(1, 2) for _ in range(8000)]
    worst = {}
    for r in (2, 4, 8):
        d = DGIM(300, buckets_per_size=r)
        e = ExactWindow(300)
        w = 0.0
        for b in stream:
            d.push(b)
            e.push(b)
            ex = e.count()
            if ex > 0:
                w = max(w, abs(d.count() - ex) / ex)
        worst[r] = w
    check("higher buckets-per-size reduces error", worst[8] <= worst[4] <= worst[2] + 1e-9,
          f"r=2:{worst[2]:.3f} r=4:{worst[4]:.3f} r=8:{worst[8]:.3f}")
    check("r=8 error is small", worst[8] < 0.2, f"{worst[8]:.3f}")

    # ---- 3. sub-window queries obey the bound -----------------------------------------
    rng = LCG(9)
    d = DGIM(500, buckets_per_size=3)
    e = ExactWindow(500)
    subbad = 0
    for _ in range(5000):
        b = rng.bit(1, 2)
        d.push(b)
        e.push(b)
    for k in (50, 100, 250, 400, 500):
        est = d.count(k)
        exact = e.count(k)
        if exact > 0 and abs(est - exact) / exact > 0.5 + 1e-9:
            subbad += 1
    check("sub-window queries within error bound", subbad == 0, f"{subbad} out of band")

    # ---- 4. logarithmic memory --------------------------------------------------------
    rng = LCG(5)
    d = DGIM(1 << 16, buckets_per_size=2)   # 65536-bit window
    for _ in range(500000):
        d.push(rng.bit(1, 2))
    n = d.window
    # O(log^2 N): bound generously
    log2n = math.log2(n)
    check("bucket count is logarithmic", d.bucket_count() <= 3 * log2n * log2n,
          f"{d.bucket_count()} buckets for window {n} (log^2={log2n*log2n:.0f})")

    # ---- 5. edge cases ----------------------------------------------------------------
    # all zeros -> exactly 0
    d = DGIM(50)
    for _ in range(200):
        d.push(0)
    check("all zeros -> count 0", d.count() == 0)

    # all ones -> within bound of window size
    d = DGIM(50)
    e = ExactWindow(50)
    for _ in range(200):
        d.push(1)
        e.push(1)
    check("all ones -> within error of full window", abs(d.count() - e.count()) / e.count() <= 0.5)

    # window larger than stream so far: DGIM should be reasonable
    d = DGIM(1000)
    e = ExactWindow(1000)
    rng = LCG(3)
    for _ in range(30):     # fewer bits than the window
        b = rng.bit(1, 2)
        d.push(b)
        e.push(b)
    exact = e.count()
    if exact > 0:
        check("short stream within error bound", abs(d.count() - exact) / exact <= 0.5,
              f"est {d.count()} exact {exact}")
    else:
        check("short stream within error bound", d.count() == 0)

    # ---- 6. parameter validation ------------------------------------------------------
    try:
        DGIM(0)
        check("window < 1 rejected", False)
    except ValueError:
        check("window < 1 rejected", True)
    try:
        DGIM(10, buckets_per_size=1)
        check("buckets_per_size < 2 rejected", False)
    except ValueError:
        check("buckets_per_size < 2 rejected", True)

    check("error_bound matches 1/(r-1)", abs(DGIM(10, 3).error_bound() - 0.5) < 1e-9)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
