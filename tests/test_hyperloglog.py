"""Tests for hyperloglog.py -- streaming cardinality estimation.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Estimates are checked against
true cardinalities to within a few standard errors.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hyperloglog as H  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- construction & parameters ---------------------------------------------
hll = H.HyperLogLog(12)
check("p=12 gives 4096 registers", hll.m == 4096)
check("registers start empty", hll.registers.count(0) == hll.m)
check("standard error is 1.04/sqrt(m)", abs(hll.standard_error() - 1.04 / math.sqrt(4096)) < 1e-9)
check("more registers -> smaller error", H.HyperLogLog(14).standard_error() < H.HyperLogLog(10).standard_error())
try:
    H.HyperLogLog(2)
    check("rejects p out of range", False)
except ValueError:
    check("rejects p out of range", True)

# --- empty and tiny ---------------------------------------------------------
check("empty sketch estimates ~0", H.HyperLogLog(12).estimate() < 1.0)
solo = H.HyperLogLog(12)
solo.add("only-one")
check("one item estimates ~1", abs(solo.estimate() - 1) < 1.0)

# --- accuracy across cardinalities -----------------------------------------
for p, n in ((12, 1000), (12, 10000), (14, 100000)):
    hll = H.HyperLogLog(p)
    for i in range(n):
        hll.add(f"item-{i}")
    est = hll.estimate()
    err = abs(H.relative_error(est, n))
    se = hll.standard_error()
    check(f"estimate within 3 SE for n={n}, p={p} (err {err:.4f} vs SE {se:.4f})", err < 3 * se)

# --- duplicates do not inflate the count -----------------------------------
dup = H.HyperLogLog(12)
for _ in range(20):
    for i in range(1000):
        dup.add(f"d-{i}")
check("counting distinct ignores duplicates", abs(H.relative_error(dup.estimate(), 1000)) < 0.1)

# --- small-range (linear counting) correction ------------------------------
small = H.HyperLogLog(12)
for i in range(100):
    small.add(f"s-{i}")
check("small cardinality is accurate (linear counting)", abs(H.relative_error(small.estimate(), 100)) < 0.1)

# --- merge is the union -----------------------------------------------------
a = H.HyperLogLog(12)
b = H.HyperLogLog(12)
for i in range(5000):
    a.add(f"a-{i}")
for i in range(2500, 7500):  # 2500 overlap -> union is 7500
    b.add(f"a-{i}")
u = a.merge(b)
check("merge estimates the union cardinality", abs(H.relative_error(u.estimate(), 7500)) < 0.05)
check("merge does not mutate the inputs",
      a.estimate() < 6000 and b.estimate() < 6000)
# merging a sketch with itself changes nothing
same = a.merge(a)
check("merging with self is idempotent-ish", abs(same.estimate() - a.estimate()) < 1e-6)
try:
    a.merge(H.HyperLogLog(10))
    check("rejects merging different precisions", False)
except ValueError:
    check("rejects merging different precisions", True)

# --- register mechanics -----------------------------------------------------
check("leading zeros of 0 is the full width", H.HyperLogLog._leading_zeros(0, 64) == 64)
check("leading zeros counts correctly", H.HyperLogLog._leading_zeros(1, 8) == 7)
check("leading zeros of top bit is 0", H.HyperLogLog._leading_zeros(1 << 7, 8) == 0)
# adding raises registers monotonically (never lowers a register)
hll = H.HyperLogLog(8)
before = bytes(hll.registers)
for i in range(500):
    hll.add(f"m-{i}")
check("registers only ever increase", all(hll.registers[i] >= before[i] for i in range(hll.m)))
check("max register is a small integer (a rank)", max(hll.registers) < 64)

# --- hash quality: registers well spread -----------------------------------
hll = H.HyperLogLog(10)
for i in range(50000):
    hll.add(f"h-{i}")
# with 50k items over 1024 registers, essentially none should remain empty
check("the finalized hash spreads across registers (few empty)", hll.registers.count(0) < 5)

# --- relative_error helper --------------------------------------------------
check("relative_error is signed", H.relative_error(110, 100) > 0 and H.relative_error(90, 100) < 0)
check("relative_error of exact is 0", H.relative_error(100, 100) == 0.0)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall hyperloglog tests passed")
