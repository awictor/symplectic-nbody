"""Tests for minhash: Jaccard estimation accuracy, edge cases, and LSH recall."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minhash import (MinHash, estimate_jaccard, true_jaccard, LSH, lsh_threshold)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 777
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- true_jaccard sanity ---------------------------------------------------
check("true jaccard of identical sets is 1", true_jaccard({1, 2, 3}, {1, 2, 3}) == 1.0)
check("true jaccard of disjoint sets is 0", true_jaccard({1, 2}, {3, 4}) == 0.0)
check("true jaccard half-overlap", abs(true_jaccard({1, 2, 3, 4}, {3, 4, 5, 6}) - 2 / 6) < 1e-12)
check("true jaccard of two empty sets is 0", true_jaccard(set(), set()) == 0.0)

# --- identical sets estimate 1, disjoint estimate ~0 -----------------------
mh = MinHash(num_hashes=256, seed=1)
a = set(range(100))
sig_a = mh.signature(a)
check("identical signatures estimate 1", estimate_jaccard(sig_a, mh.signature(set(range(100)))) == 1.0)

disj_b = mh.signature(set(range(1000, 1100)))
check("disjoint sets estimate near 0", estimate_jaccard(sig_a, disj_b) < 0.05)

# --- symmetry --------------------------------------------------------------
s1 = mh.signature({1, 2, 3, 4, 5, 6})
s2 = mh.signature({4, 5, 6, 7, 8, 9})
check("estimator is symmetric", estimate_jaccard(s1, s2) == estimate_jaccard(s2, s1))

# --- estimate converges to true Jaccard as k grows -------------------------
# build many random set pairs with a spread of true Jaccard, measure mean abs error
def random_pair():
    base = set(int(rng() * 200) for _ in range(60))
    # overlap fraction random
    keep = [e for e in base if rng() < 0.5 + 0.5 * rng()]
    other = set(keep) | set(int(rng() * 200) for _ in range(40))
    return base, other


for k, tol in [(64, 0.06), (256, 0.03)]:
    mh_k = MinHash(num_hashes=k, seed=7)
    total_err = 0.0
    trials = 80
    for _ in range(trials):
        A, B = random_pair()
        est = estimate_jaccard(mh_k.signature(A), mh_k.signature(B))
        total_err += abs(est - true_jaccard(A, B))
    mean_err = total_err / trials
    check(f"mean |estimate - true| < {tol} at k={k} (got {mean_err:.4f})", mean_err < tol)

# --- error shrinks as k increases ------------------------------------------
def mean_error(k):
    mh_k = MinHash(num_hashes=k, seed=11)
    tot = 0.0
    T = 60
    for _ in range(T):
        A, B = random_pair()
        tot += abs(estimate_jaccard(mh_k.signature(A), mh_k.signature(B)) - true_jaccard(A, B))
    return tot / T


e16 = mean_error(16)
e256 = mean_error(256)
check(f"error decreases with more hashes ({e16:.4f} -> {e256:.4f})", e256 < e16)

# --- empty set handling ----------------------------------------------------
empty_sig = mh.signature(set())
check("empty-set signature vs itself estimates 0", estimate_jaccard(empty_sig, empty_sig) == 0.0)
check("empty vs non-empty estimates 0", estimate_jaccard(empty_sig, sig_a) == 0.0)

# --- deterministic across instances with the same seed ---------------------
mh_x = MinHash(num_hashes=64, seed=42)
mh_y = MinHash(num_hashes=64, seed=42)
check("same seed -> identical signatures", mh_x.signature({1, 2, 3}) == mh_y.signature({1, 2, 3}))

# --- LSH threshold formula -------------------------------------------------
t = lsh_threshold(20, 5)
check("lsh_threshold in (0,1)", 0 < t < 1)
check("more rows -> higher threshold", lsh_threshold(20, 10) > lsh_threshold(20, 2))

# --- LSH recalls similar pairs, filters dissimilar -------------------------
# make clusters of near-duplicate sets and some unrelated sets
mh_l = MinHash(num_hashes=100, seed=3)
bands, rows = 25, 4       # 25*4 = 100
lsh = LSH(bands, rows)

items = {}
# 5 near-duplicate groups: each group shares ~90% of elements
group_bases = []
for g in range(5):
    base = set(int(rng() * 500) for _ in range(80))
    group_bases.append(base)
    for v in range(4):
        variant = set(e for e in base if rng() < 0.95) | set(int(rng() * 500) for _ in range(5))
        key = f"g{g}_v{v}"
        items[key] = variant
        lsh.add(key, mh_l.signature(variant))

# 10 unrelated random sets
for u in range(10):
    key = f"rand{u}"
    s = set(int(rng() * 5000) for _ in range(80))
    items[key] = s
    lsh.add(key, mh_l.signature(s))

cand_pairs = lsh.all_candidate_pairs()

# every genuinely-similar within-group pair (true Jaccard > 0.5) should be a candidate
missed = 0
same_group_similar = 0
for g in range(5):
    keys = [f"g{g}_v{v}" for v in range(4)]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            tj = true_jaccard(items[keys[i]], items[keys[j]])
            if tj > 0.5:
                same_group_similar += 1
                pair = tuple(sorted((keys[i], keys[j])))
                if pair not in cand_pairs and pair[::-1] not in cand_pairs:
                    missed += 1
check(f"LSH recalls all high-Jaccard within-group pairs ({same_group_similar - missed}/{same_group_similar})",
      missed == 0)

# candidate pairs should be dominated by similar ones (few false positives among random sets)
false_pos = 0
for (x, y) in cand_pairs:
    if true_jaccard(items[x], items[y]) < 0.2:
        false_pos += 1
check(f"LSH keeps false positives low ({false_pos} dissimilar candidate pairs)", false_pos <= 3)

# candidates() for a query returns items sharing a band
q = items["g0_v0"]
cands = lsh.candidates(mh_l.signature(q))
check("query candidates include same-group variants",
      any(c.startswith("g0_") for c in cands))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all minhash tests passed")
