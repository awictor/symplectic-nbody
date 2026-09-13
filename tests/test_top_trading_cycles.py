"""Tests for top_trading_cycles: permutation, individual rationality, Pareto, core, strategy-proof."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from top_trading_cycles import (top_trading_cycles, is_permutation,  # noqa: E402
                                is_individually_rational, is_pareto_efficient, in_core)


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

    def perm(self, n):
        o = list(range(n))
        for k in range(n - 1, 0, -1):
            j = self.randint(0, k)
            o[k], o[j] = o[j], o[k]
        return o


def main():
    # ---- 1. everyone prefers their own object -> identity allocation ------------------
    n = 5
    owner = list(range(n))
    prefs = [[i] + [j for j in range(n) if j != i] for i in range(n)]
    alloc, rounds = top_trading_cycles(owner, prefs)
    check("all-own-favourite -> identity", alloc == list(range(n)))
    check("identity clears in one round of self-loops", len(rounds) == 1)

    # ---- 2. one big cycle: everyone wants the next object -----------------------------
    prefs = [[(i + 1) % n] + [j for j in range(n) if j != (i + 1) % n] for i in range(n)]
    alloc, rounds = top_trading_cycles(owner, prefs)
    check("big-cycle preferences shift everyone around", alloc == [(i + 1) % n for i in range(n)])
    check("big cycle clears in one round", len(rounds) == 1 and len(rounds[0]) == 1)

    # ---- 3. a two-person swap -------------------------------------------------------
    #  person 0 wants object 1, person 1 wants object 0, others keep theirs
    prefs = [[1, 0, 2, 3], [0, 1, 2, 3], [2, 0, 1, 3], [3, 0, 1, 2]]
    alloc, _ = top_trading_cycles([0, 1, 2, 3], prefs)
    check("mutual swap executes", alloc[0] == 1 and alloc[1] == 0 and alloc[2] == 2 and alloc[3] == 3)

    # ---- 4. random instances satisfy all the core properties -------------------------
    rng = LCG(2024)
    bad_perm = bad_ir = bad_pareto = bad_core = 0
    for _ in range(400):
        n = rng.randint(2, 6)
        owner = list(range(n))
        prefs = [rng.perm(n) for _ in range(n)]
        alloc, _ = top_trading_cycles(owner, prefs)
        if not is_permutation(alloc):
            bad_perm += 1
        if not is_individually_rational(owner, prefs, alloc):
            bad_ir += 1
        if not is_pareto_efficient(prefs, alloc):
            bad_pareto += 1
        if not in_core(owner, prefs, alloc):
            bad_core += 1
    check("output is always a permutation", bad_perm == 0, f"{bad_perm}")
    check("always individually rational", bad_ir == 0, f"{bad_ir}")
    check("always Pareto efficient", bad_pareto == 0, f"{bad_pareto}")
    check("always in the core", bad_core == 0, f"{bad_core}")

    # ---- 5. arbitrary ownership (not identity) ---------------------------------------
    # person i owns object (i+2) mod n
    n = 4
    owner = [(k - 2) % n for k in range(n)]   # owner[k] = who owns object k
    # rebuild so owner[k] is holder of object k: pick owner as a permutation
    owner = [2, 3, 0, 1]                       # object k owned by owner[k]
    prefs = [rng.perm(n) for _ in range(n)]
    alloc, _ = top_trading_cycles(owner, prefs)
    check("arbitrary ownership: valid permutation", is_permutation(alloc))
    check("arbitrary ownership: individually rational", is_individually_rational(owner, prefs, alloc))
    check("arbitrary ownership: in core", in_core(owner, prefs, alloc))

    # ---- 6. strategy-proofness on sampled unilateral misreports -----------------------
    # no person can get a strictly better object by lying about their preferences
    rng = LCG(77)
    sp_violations = 0
    for _ in range(60):
        n = rng.randint(2, 5)
        owner = list(range(n))
        prefs = [rng.perm(n) for _ in range(n)]
        alloc, _ = top_trading_cycles(owner, prefs)
        rank = [{o: pos for pos, o in enumerate(prefs[i])} for i in range(n)]
        for liar in range(n):
            truthful_obj = alloc[liar]
            # try every possible false report for `liar`
            for _ in range(10):
                fake = rng.perm(n)
                fake_prefs = [list(p) for p in prefs]
                fake_prefs[liar] = fake
                a2, _ = top_trading_cycles(owner, fake_prefs)
                # did lying get the liar a genuinely more-preferred object (by TRUE prefs)?
                if rank[liar][a2[liar]] < rank[liar][truthful_obj]:
                    sp_violations += 1
    check("strategy-proof: lying never yields a better object (sampled)", sp_violations == 0,
          f"{sp_violations} violations")

    # ---- 7. rounds partition the population -------------------------------------------
    n = 6
    owner = list(range(n))
    prefs = [rng.perm(n) for _ in range(n)]
    alloc, rounds = top_trading_cycles(owner, prefs)
    all_in_rounds = sorted(p for rnd in rounds for cyc in rnd for p in cyc)
    check("every person cleared in exactly one cycle", all_in_rounds == list(range(n)))

    # ---- 8. n=1 trivial ---------------------------------------------------------------
    alloc, _ = top_trading_cycles([0], [[0]])
    check("single person keeps own object", alloc == [0])

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
