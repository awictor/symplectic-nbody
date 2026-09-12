"""Tests for cuckoo_filter: no false negatives, low false positives, deletion, load factor."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cuckoo_filter import CuckooFilter

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- no false negatives: every added item tests present --------------------
cf = CuckooFilter(capacity=4096, bucket_size=4, fingerprint_bits=16, seed=1)
items = [f"item-{i}" for i in range(1000)]
added = [x for x in items if cf.add(x)]
check("most items insert successfully", len(added) >= 990)
check("no false negatives: every added item is present", all(x in cf for x in added))

# --- false-positive rate is small ------------------------------------------
absent = [f"absent-{i}" for i in range(20000)]
fp = sum(1 for x in absent if x in cf)
rate = fp / len(absent)
# with 16-bit fingerprints and 2 buckets, the FP rate is roughly 2*bucket_size/2^16 ~ 1.2e-4
check(f"false-positive rate is small ({rate:.5f})", rate < 0.01)

# --- deletion removes an item ----------------------------------------------
cf2 = CuckooFilter(capacity=1024, seed=2)
for i in range(200):
    cf2.add(f"k{i}")
check("item present before deletion", "k100" in cf2)
cf2.delete("k100")
# it should be absent now (unless a fingerprint collision keeps it; with 16 bits, unlikely)
check("item absent after deletion", "k100" not in cf2)
# other items unaffected
check("other items survive deletion", all(f"k{i}" in cf2 for i in range(200) if i != 100))

# --- deleting a non-member returns False -----------------------------------
check("deleting a non-member returns False", not cf2.delete("never-added"))

# --- duplicates: adding an item twice, deleting once leaves one ------------
cf3 = CuckooFilter(capacity=256, seed=3)
cf3.add("dup")
cf3.add("dup")
check("duplicate added twice is present", "dup" in cf3)
cf3.delete("dup")
check("after deleting one duplicate, still present", "dup" in cf3)
cf3.delete("dup")
check("after deleting both duplicates, absent", "dup" not in cf3)

# --- the count tracks additions and deletions ------------------------------
cf4 = CuckooFilter(capacity=512, seed=4)
for i in range(100):
    cf4.add(f"c{i}")
check("count reflects additions", cf4.count == 100)
cf4.delete("c0")
cf4.delete("c1")
check("count reflects deletions", cf4.count == 98)

# --- load factor stays below 1 ---------------------------------------------
cf5 = CuckooFilter(capacity=1024, bucket_size=4, seed=5)
for i in range(2000):
    cf5.add(f"x{i}")
check("load factor is at most 1", cf5.load_factor() <= 1.0)
check("load factor is high (efficient packing)", cf5.load_factor() > 0.5)

# --- reproducibility -------------------------------------------------------
def build(seed):
    cf = CuckooFilter(capacity=512, seed=seed)
    for i in range(200):
        cf.add(f"r{i}")
    return [f"r{i}" in cf for i in range(300)]
check("same seed gives identical membership results", build(7) == build(7))

# --- integer and bytes keys ------------------------------------------------
cf6 = CuckooFilter(capacity=256, seed=1)
cf6.add(12345)
cf6.add(b"raw-bytes")
check("integer key round-trips", 12345 in cf6)
check("bytes key round-trips", b"raw-bytes" in cf6)
check("a different integer is (almost surely) absent", 999999 not in cf6)

# --- adds up to a good load factor without failing early -------------------
cf7 = CuckooFilter(capacity=1000, bucket_size=4, seed=8)
success = sum(cf7.add(f"load{i}") for i in range(950))
check(f"inserts most items up to ~95% capacity ({success}/950)", success >= 900)

# --- everything still present after heavy load -----------------------------
present = sum(1 for i in range(950) if f"load{i}" in cf7)
check("all inserted items remain present under load", present >= success)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all cuckoo_filter tests passed")
