"""Tests for kahan.py -- compensated summation.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Compensated sums are checked
against Python's exact math.fsum on ill-conditioned inputs; the naive sum's error is shown.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import kahan as K  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- exactness on simple inputs --------------------------------------------
check("empty sum is 0", K.kahan_sum([]) == 0.0)
check("single value", K.kahan_sum([3.5]) == 3.5)
check("integers sum exactly", K.kahan_sum([1, 2, 3, 4, 5]) == 15.0)
check("naive and kahan agree on small exact sums", K.naive_sum([1, 2, 3]) == K.kahan_sum([1, 2, 3]))

# --- the classic 0.1 * n accumulation --------------------------------------
vals = [0.1] * 1000000
exact = math.fsum(vals)                 # 100000.0, computed exactly
check("Kahan matches fsum on a million 0.1s", K.kahan_sum(vals) == exact)
check("Neumaier matches fsum on a million 0.1s", K.neumaier_sum(vals) == exact)
check("the naive sum has visible error", K.relative_error(K.naive_sum(vals), exact) > 1e-12)
check("Kahan error is far smaller than naive",
      K.relative_error(K.kahan_sum(vals), exact) < K.relative_error(K.naive_sum(vals), exact) / 100)
check("pairwise sum is much more accurate than naive",
      K.relative_error(K.pairwise_sum(vals), exact) < K.relative_error(K.naive_sum(vals), exact) / 100)

# --- ill-conditioned sums (large cancellation) -----------------------------
ill = [1.0, 1e100, 1.0, -1e100]         # exact answer is 2.0
check("Neumaier handles catastrophic cancellation (= 2)", K.neumaier_sum(ill) == 2.0)
check("Neumaier matches fsum on the ill-conditioned sum", K.neumaier_sum(ill) == math.fsum(ill))
check("the naive sum loses both small terms (= 0)", K.naive_sum(ill) == 0.0)
# a longer alternating-large sequence
big = ([1e16, 1.0, -1e16, 1.0] * 1000)
check("Neumaier recovers a long cancellation sum", K.neumaier_sum(big) == math.fsum(big))

# --- random ill-conditioned data vs fsum -----------------------------------
def lcg(seed):
    s = seed
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield s >> 8


gen = lcg(1)
kahan_ok = neumaier_ok = True
for _ in range(50):
    n = 500 + next(gen) % 2000
    # mix of large and tiny magnitudes to stress the summation
    data = []
    for _ in range(n):
        mag = 10 ** ((next(gen) % 20) - 10)     # 1e-10 .. 1e9
        sign = 1 if next(gen) % 2 else -1
        data.append(sign * mag * (0.5 + (next(gen) / (1 << 24))))
    fs = math.fsum(data)
    if K.relative_error(K.kahan_sum(data), fs) > 1e-9 and abs(fs) > 1e-6:
        kahan_ok = False
    if abs(fs) > 1e-6 and K.relative_error(K.neumaier_sum(data), fs) > 1e-10:
        neumaier_ok = False
check("Kahan stays close to fsum on 50 random mixed-magnitude sums", kahan_ok)
check("Neumaier stays very close to fsum on the same", neumaier_ok)

# --- compensated dot product -----------------------------------------------
u = [0.1] * 100000
v = [0.1] * 100000
exact_dot = math.fsum(a * b for a, b in zip(u, v))
check("kahan_dot matches the exact dot product", K.kahan_dot(u, v) == exact_dot)
check("kahan_dot on orthogonal-ish data", abs(K.kahan_dot([1, 0, 1], [0, 1, 0])) == 0.0)
try:
    K.kahan_dot([1, 2], [1])
    check("kahan_dot rejects mismatched lengths", False)
except ValueError:
    check("kahan_dot rejects mismatched lengths", True)

# --- streaming accumulator --------------------------------------------------
acc = K.KahanAccumulator()
for _ in range(1000000):
    acc.add(0.1)
check("accumulator sum matches exact (1e6 * 0.1)", acc.sum() == 100000.0)
check("accumulator mean is exactly 0.1", acc.mean() == 0.1)
check("accumulator tracks the count", acc.n == 1000000)
check("empty accumulator mean is 0", K.KahanAccumulator().mean() == 0.0)

# --- pairwise sum correctness ----------------------------------------------
check("pairwise sum of a small list is exact", K.pairwise_sum([1, 2, 3, 4]) == 10.0)
check("pairwise sum matches naive on integers", K.pairwise_sum(list(range(1000))) == sum(range(1000)))

# --- relative_error helper --------------------------------------------------
check("relative_error is 0 for an exact match", K.relative_error(2.0, 2.0) == 0.0)
check("relative_error handles exact=0", K.relative_error(1e-9, 0.0) == 1e-9)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall kahan tests passed")
