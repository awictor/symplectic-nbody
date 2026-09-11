"""Tests for bloom.py -- Bloom filter probabilistic set membership.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. The central guarantees are no
false negatives and a false-positive rate matching theory.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import bloom  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- optimal parameters -----------------------------------------------------
m = bloom.optimal_num_bits(1000, 0.01)
check("1000 items @1% needs ~9586 bits", 9500 <= m <= 9700)
check("bits per item is ~9.6 for 1% error", approx(m / 1000, 9.6, 0.2))
k = bloom.optimal_num_hashes(m, 1000)
check("optimal hashes for that size is ~7", k == 7)
check("more items at fixed rate needs more bits",
      bloom.optimal_num_bits(2000, 0.01) > bloom.optimal_num_bits(1000, 0.01))
check("lower error rate needs more bits",
      bloom.optimal_num_bits(1000, 0.001) > bloom.optimal_num_bits(1000, 0.01))
try:
    bloom.optimal_num_bits(1000, 1.5)
    check("rejects invalid rate", False)
except ValueError:
    check("rejects invalid rate", True)

# --- theoretical false-positive formula ------------------------------------
check("empty filter (n=0) has zero false-positive rate", bloom.false_positive_rate(1000, 0, 5) == 0.0)
check("fp rate rises as more items are inserted",
      bloom.false_positive_rate(9586, 2000, 7) > bloom.false_positive_rate(9586, 1000, 7))
check("optimal k gives ~ the target rate", approx(bloom.false_positive_rate(m, 1000, k), 0.01, 0.005))
# optimal k minimizes the false-positive rate: neighbours are no better
base = bloom.false_positive_rate(m, 1000, k)
check("optimal k is a local minimum",
      base <= bloom.false_positive_rate(m, 1000, k - 1) and base <= bloom.false_positive_rate(m, 1000, k + 1))

# --- construction -----------------------------------------------------------
bf = bloom.BloomFilter.for_capacity(1000, 0.01)
check("for_capacity sizes the bit array", 9500 <= bf.num_bits <= 9700)
check("for_capacity picks the optimal hashes", bf.num_hashes == 7)
check("a fresh filter has no bits set", bf.bits_set() == 0)
try:
    bloom.BloomFilter(0, 3)
    check("rejects zero bits", False)
except ValueError:
    check("rejects zero bits", True)

# --- the core guarantees ----------------------------------------------------
items = [f"item-{i}" for i in range(1000)]
for x in items:
    bf.add(x)
# NO FALSE NEGATIVES: everything added must test present
false_negatives = sum(1 for x in items if x not in bf)
check("no false negatives -- every added item is present", false_negatives == 0)
check("bits are set after adding", bf.bits_set() > 0)
check("fill ratio is about ln 2 at optimal load", approx(bf.fill_ratio(), 0.5, 0.06))

# observed false-positive rate on never-added items matches theory
never = [f"absent-{i}" for i in range(20000)]
fp = sum(1 for x in never if x in bf)
observed = fp / len(never)
theoretical = bloom.false_positive_rate(bf.num_bits, 1000, bf.num_hashes)
check("observed FP rate is close to theory", approx(observed, theoretical, 0.01))
check("observed FP rate is near the 1% target", observed < 0.03)
check("estimated FP rate (from fill) tracks observed",
      approx(bf.estimated_false_positive_rate(), observed, 0.01))

# --- different item types ---------------------------------------------------
bf2 = bloom.BloomFilter(1000, 4)
bf2.add(42)
bf2.add(b"raw bytes")
bf2.add((1, 2, 3))
check("integer membership works", 42 in bf2)
check("bytes membership works", b"raw bytes" in bf2)
check("tuple membership works", (1, 2, 3) in bf2)
check("an unseen item is (almost surely) absent", "definitely-not-added-xyz" not in bf2)

# --- smaller filter, higher error rate, still no false negatives -----------
small = bloom.BloomFilter(200, 3)
words = ["apple", "banana", "cherry", "date", "elderberry"]
for w in words:
    small.add(w)
check("small filter has no false negatives", all(w in small for w in words))
check("add increments the count", small.count == len(words))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall bloom tests passed")
