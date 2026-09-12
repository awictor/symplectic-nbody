"""Tests for gale_shapley: stable matching stability, proposer-optimality vs brute enumeration."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gale_shapley import (stable_matching, blocking_pairs, is_stable,
                          reviewer_optimal_matching, all_stable_matchings)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def rand(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s >> 16

    def randint(self, lo, hi):
        return lo + self.rand() % (hi - lo + 1)


def random_prefs(rng, n):
    prefs = []
    for _ in range(n):
        order = list(range(n))
        for i in range(n - 1, 0, -1):
            j = rng.rand() % (i + 1)
            order[i], order[j] = order[j], order[i]
        prefs.append(order)
    return prefs


def rank_of(prefs, who, target):
    return prefs[who].index(target)


# --- known case -------------------------------------------------------------
pp = [[0, 1, 2], [1, 0, 2], [0, 1, 2]]
rp = [[1, 0, 2], [0, 1, 2], [0, 1, 2]]
m = stable_matching(3, pp, rp)
check("known 3x3: matching is stable", is_stable(3, pp, rp, m))
check("known 3x3: proposer-optimal matching is [0,1,2]", m == [0, 1, 2])
check("known 3x3: exactly two stable matchings", len(all_stable_matchings(3, pp, rp)) == 2)

# everyone-agrees case: identical preferences -> unique stable matching (the "diagonal")
pp = [[0, 1, 2, 3]] * 4
rp = [[0, 1, 2, 3]] * 4
m = stable_matching(4, pp, rp)
check("identical prefs: proposer 0 gets reviewer 0", m[0] == 0)
check("identical prefs: unique stable matching", len(all_stable_matchings(4, pp, rp)) == 1)

# --- the returned matching is always stable and complete -------------------
rng = LCG(2026)
stable_ok = True
for _ in range(500):
    n = rng.randint(1, 7)
    pp = random_prefs(rng, n)
    rp = random_prefs(rng, n)
    m = stable_matching(n, pp, rp)
    if sorted(m) != list(range(n)):          # a valid perfect matching
        stable_ok = False
        break
    if not is_stable(n, pp, rp, m):
        stable_ok = False
        print(f"  unstable result: pp={pp} rp={rp} m={m} blocks={blocking_pairs(n, pp, rp, m)}")
        break
check("Gale-Shapley always returns a complete, stable matching (500 profiles)", stable_ok)

# --- proposer-optimality vs brute ------------------------------------------
rng = LCG(4242)
opt_ok = True
for _ in range(300):
    n = rng.randint(1, 6)
    pp = random_prefs(rng, n)
    rp = random_prefs(rng, n)
    m = stable_matching(n, pp, rp)
    stables = all_stable_matchings(n, pp, rp)
    # every stable matching must be found by brute (GS result among them)
    if tuple(m) not in stables:
        opt_ok = False
        break
    # proposer-optimal: for every proposer, no stable matching gives a strictly better partner
    for p in range(n):
        gs_rank = rank_of(pp, p, m[p])
        for sm in stables:
            if rank_of(pp, p, sm[p]) < gs_rank:
                opt_ok = False
                break
        if not opt_ok:
            break
    if not opt_ok:
        break
check("proposer-optimal: no proposer beats his GS partner in any stable matching (300 profiles)",
      opt_ok)

# --- reviewer-optimal is reviewer-optimal ----------------------------------
rng = LCG(777)
rev_ok = True
for _ in range(300):
    n = rng.randint(1, 6)
    pp = random_prefs(rng, n)
    rp = random_prefs(rng, n)
    rm = reviewer_optimal_matching(n, pp, rp)
    if not is_stable(n, pp, rp, rm):
        rev_ok = False
        break
    # build reviewer->proposer for the reviewer-optimal matching
    match_r = [-1] * n
    for p in range(n):
        match_r[rm[p]] = p
    stables = all_stable_matchings(n, pp, rp)
    for r in range(n):
        rev_rank = rank_of(rp, r, match_r[r])
        for sm in stables:
            # reviewer r's partner in sm
            sm_partner = next(p for p in range(n) if sm[p] == r)
            if rank_of(rp, r, sm_partner) < rev_rank:
                rev_ok = False
                break
        if not rev_ok:
            break
    if not rev_ok:
        break
check("reviewer-optimal: no reviewer beats her partner in any stable matching (300 profiles)", rev_ok)

# --- unique stable matching -> proposer- and reviewer-optimal agree --------
rng = LCG(555)
unique_ok = True
tested = 0
for _ in range(300):
    n = rng.randint(1, 6)
    pp = random_prefs(rng, n)
    rp = random_prefs(rng, n)
    if len(all_stable_matchings(n, pp, rp)) == 1:
        if stable_matching(n, pp, rp) != reviewer_optimal_matching(n, pp, rp):
            unique_ok = False
            break
        tested += 1
check(f"unique stable matching -> proposer- and reviewer-optimal agree ({tested} profiles)", unique_ok)

# --- blocking-pair detection is correct ------------------------------------
# a deliberately unstable matching: swap two partners that would block
pp = [[0, 1], [0, 1]]
rp = [[1, 0], [1, 0]]
# match proposer0->reviewer0, proposer1->reviewer1
# reviewer0 prefers proposer1, proposer1 prefers reviewer0 -> (1,0) blocks
bad = [0, 1]
check("blocking pair detected in an unstable matching",
      (1, 0) in blocking_pairs(2, pp, rp, bad) and not is_stable(2, pp, rp, bad))

# and Gale-Shapley fixes it
good = stable_matching(2, pp, rp)
check("Gale-Shapley returns the stable matching for that instance", is_stable(2, pp, rp, good))

# --- larger instance solves and is stable ----------------------------------
rng = LCG(31337)
n = 200
pp = random_prefs(rng, n)
rp = random_prefs(rng, n)
m = stable_matching(n, pp, rp)
check("200x200 instance: matching is complete and stable",
      sorted(m) == list(range(n)) and is_stable(n, pp, rp, m))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all gale_shapley tests passed")
