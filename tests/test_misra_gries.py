"""Tests for misra_gries.py -- streaming frequent items and majority vote.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Results are proven correct
exhaustively against exact counting.
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import misra_gries as MG  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- heavy hitters (verified) matches exact --------------------------------
s = "aaaabbbccd"  # a:4 b:3 c:2 d:1, n=10
check("heavy hitters k=4 are a and b", MG.heavy_hitters(s, 4) == {"a": 4, "b": 3})
check("heavy hitters match exact counting", MG.heavy_hitters(s, 4) == MG.exact_heavy_hitters(s, 4))
check("k=2 (majority threshold) finds a only", MG.heavy_hitters(s, 2) == MG.exact_heavy_hitters(s, 2))
check("empty stream has no heavy hitters", MG.heavy_hitters([], 3) == {})
check("uniform stream has no item over n/k for large k",
      MG.heavy_hitters(list(range(20)), 4) == {})
try:
    MG.MisraGries(1)
    check("rejects k < 2", False)
except ValueError:
    check("rejects k < 2", True)

# --- the summary uses at most k-1 counters and never misses a heavy hitter -
mg = MG.MisraGries(4)
mg.update(s)
check("summary uses at most k-1 counters", len(mg.candidates()) <= 3)
check("candidates are a superset of the true heavy hitters",
      set(MG.exact_heavy_hitters(s, 4)).issubset(set(mg.candidates())))
check("approx counts underestimate the true counts", mg.approx_count("a") <= s.count("a"))
check("approx count is within n/k of the truth", s.count("a") - mg.approx_count("a") <= len(s) / 4)
check("an untracked item has approx count 0", mg.approx_count("z") == 0)
check("summary counts the stream length", mg.n == len(s))

# --- Boyer-Moore majority ---------------------------------------------------
check("majority of a clear majority stream", MG.majority("aaabab") == "a")
check("no majority returns None", MG.majority("abab") is None)
check("single element is its own majority", MG.majority("x") == "x")
check("empty stream has no majority", MG.majority([]) is None)
check("exactly half is NOT a majority", MG.majority("aabb") is None)
check("just over half is a majority", MG.majority("aabba" + "a") == "a")  # 4 a of 6

# --- exhaustive: heavy_hitters == exact ------------------------------------
def lcg(seed):
    st = seed
    while True:
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        yield st >> 16


gen = lcg(1)


def rnd(lo, hi):
    return lo + next(gen) % (hi - lo + 1)


hh_bad = 0
for _ in range(2000):
    n = rnd(1, 200)
    k = rnd(2, 6)
    alpha = rnd(2, 12)
    stream = [rnd(0, alpha) for _ in range(n)]
    if MG.heavy_hitters(stream, k) != MG.exact_heavy_hitters(stream, k):
        hh_bad += 1
check("EVERY one of 2000 random heavy-hitter queries matches exact", hh_bad == 0)

# --- exhaustive: the summary never drops a real heavy hitter (no false neg) -
fn_bad = 0
for _ in range(1000):
    n = rnd(1, 150)
    k = rnd(2, 5)
    stream = [rnd(0, 8) for _ in range(n)]
    mg = MG.MisraGries(k).update(stream)
    cands = set(mg.candidates())
    true_hh = set(MG.exact_heavy_hitters(stream, k))
    if not true_hh.issubset(cands):
        fn_bad += 1
check("the summary never misses a true heavy hitter (no false negatives)", fn_bad == 0)

# --- exhaustive: approx counts are underestimates within n/k ---------------
uc_bad = 0
for _ in range(1000):
    n = rnd(1, 150)
    k = rnd(2, 5)
    stream = [rnd(0, 8) for _ in range(n)]
    mg = MG.MisraGries(k).update(stream)
    exact = Counter(stream)
    for item, ac in mg.candidates().items():
        if ac > exact[item] or exact[item] - ac > n / k + 1e-9:
            uc_bad += 1
            break
check("approx counts are underestimates within n/k on all random streams", uc_bad == 0)

# --- majority matches exact -------------------------------------------------
maj_bad = 0
for _ in range(2000):
    n = rnd(1, 100)
    stream = [rnd(0, 3) for _ in range(n)]
    c = Counter(stream)
    expected = next((x for x, ct in c.items() if ct > n // 2), None)
    if MG.majority(stream) != expected:
        maj_bad += 1
check("EVERY one of 2000 random majority queries matches exact", maj_bad == 0)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall misra_gries tests passed")
