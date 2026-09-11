"""Tests for count_min.py -- the Count-Min sketch.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. The core guarantees are checked
against exact counting: never underestimates, and stays within the error bound.
"""

import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import count_min as CM  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- construction & sizing --------------------------------------------------
cms = CM.CountMinSketch(100, 4)
check("dimensions are recorded", cms.width == 100 and cms.depth == 4)
check("a fresh sketch estimates 0", cms.estimate("anything") == 0)
sized = CM.CountMinSketch.from_error(0.01, 0.01)
check("from_error sizes width = ceil(e/eps)", sized.width == math.ceil(math.e / 0.01))
check("from_error sizes depth = ceil(ln(1/delta))", sized.depth == math.ceil(math.log(1 / 0.01)))
try:
    CM.CountMinSketch(0, 4)
    check("rejects nonpositive width", False)
except ValueError:
    check("rejects nonpositive width", True)
try:
    CM.CountMinSketch.from_error(2.0, 0.5)
    check("from_error rejects epsilon out of range", False)
except ValueError:
    check("from_error rejects epsilon out of range", True)

# --- basic counting ---------------------------------------------------------
cms = CM.CountMinSketch(200, 4)
for _ in range(5):
    cms.add("apple")
cms.add("banana", 3)
check("counts single adds", cms.estimate("apple") >= 5)
check("counts weighted adds", cms.estimate("banana") >= 3)
check("total tracks all weight", cms.total == 5 + 3)
check("an unseen item estimates 0 (in a small clean sketch)", cms.estimate("unseen-item-xyz") == 0)
try:
    cms.add("x", -1)
    check("rejects negative count", False)
except ValueError:
    check("rejects negative count", True)

# --- the never-underestimate guarantee (the whole point) -------------------
def lcg(seed):
    s = seed
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield s >> 16


gen = lcg(1)


def rnd(lo, hi):
    return lo + next(gen) % (hi - lo + 1)


under = 0
bound_violations = 0
for _ in range(30):
    # a skewed stream: a few hot keys plus a long tail
    stream = []
    for _ in range(2000):
        r = rnd(0, 99)
        if r < 40:
            stream.append("HOT1")
        elif r < 60:
            stream.append("HOT2")
        else:
            stream.append(f"tail{rnd(0, 400)}")
    exact = Counter(stream)
    sk = CM.CountMinSketch.from_error(0.01, 0.01).update(stream)
    eb = sk.error_bound()
    for item, true in exact.items():
        est = sk.estimate(item)
        if est < true:
            under += 1
        if est - true > eb:
            bound_violations += 1
check("NEVER underestimates across 30 skewed streams", under == 0)
check("estimates stay within the error bound e/w * total", bound_violations == 0)

# --- hot keys are estimated essentially exactly ----------------------------
stream = ["HOT"] * 5000 + [f"cold{i}" for i in range(3000)]
sk = CM.CountMinSketch.from_error(0.001, 0.01).update(stream)
check("a dominant key is estimated almost exactly", abs(sk.estimate("HOT") - 5000) <= sk.error_bound())
check("its estimate is at least the true count", sk.estimate("HOT") >= 5000)

# --- merge ------------------------------------------------------------------
a = CM.CountMinSketch(80, 4).update(["a", "a", "b"])
b = CM.CountMinSketch(80, 4).update(["a", "c"])
m = a.merge(b)
check("merge sums the counts", m.estimate("a") >= 3)
check("merge preserves other keys", m.estimate("b") >= 1 and m.estimate("c") >= 1)
check("merge sums the totals", m.total == 5)
try:
    a.merge(CM.CountMinSketch(80, 5))
    check("merge rejects mismatched shapes", False)
except ValueError:
    check("merge rejects mismatched shapes", True)
# merging is equivalent to adding both streams into one sketch
combined = CM.CountMinSketch(80, 4).update(["a", "a", "b", "a", "c"])
check("merge matches a single combined sketch", all(m.table[r] == combined.table[r] for r in range(4)))

# --- error bound and heavy hitters -----------------------------------------
big = CM.CountMinSketch.from_error(0.005, 0.01)
big.update(["A"] * 4000 + ["B"] * 2000 + [f"z{i}" for i in range(4000)])
check("error bound is e/w * total", abs(big.error_bound() - math.e / big.width * big.total) < 1e-9)
hh = big.heavy_hitters(0.1, {"A", "B"} | {f"z{i}" for i in range(50)})
check("heavy hitters finds the hot keys", "A" in hh and "B" in hh)
check("heavy hitters excludes cold keys", all(k in ("A", "B") for k in hh))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall count_min tests passed")
