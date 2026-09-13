"""Tests for permanent: three methods agree, known values, bipartite matching count vs brute force."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from permanent import (permanent_naive, permanent_ryser, permanent_glynn, permanent,  # noqa: E402
                       count_perfect_matchings, brute_count_matchings)


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


def main():
    # ---- 1. known values ---------------------------------------------------------------
    for n in range(1, 8):
        ones = [[1] * n for _ in range(n)]
        check(f"perm(all-ones {n}x{n}) = {n}!", permanent_ryser(ones) == math.factorial(n),
              f"{permanent_ryser(ones)}")
    ident = [[1 if i == j else 0 for j in range(5)] for i in range(5)]
    check("perm(identity) = 1", permanent_ryser(ident) == 1)
    # permutation matrix (single 1 per row/col) has permanent 1
    P = [[0] * 4 for _ in range(4)]
    for i, j in enumerate([2, 0, 3, 1]):
        P[i][j] = 1
    check("perm(permutation matrix) = 1", permanent_ryser(P) == 1)
    # zero row -> permanent 0
    Z = [[1, 2, 3], [0, 0, 0], [4, 5, 6]]
    check("perm(matrix with zero row) = 0", permanent_ryser(Z) == 0)
    # 2x2: ad + bc
    check("2x2 permanent = ad+bc", permanent_ryser([[2, 3], [5, 7]]) == 2 * 7 + 3 * 5)

    # ---- 2. Ryser matches the naive definition exhaustively up to n=8 -----------------
    rng = LCG(2024)
    mism = 0
    for n in range(1, 9):
        for _ in range(20):
            M = [[rng.randint(0, 6) for _ in range(n)] for _ in range(n)]
            if permanent_ryser(M) != permanent_naive(M):
                mism += 1
    check("Ryser == naive definition (n<=8, 160 matrices)", mism == 0, f"{mism} mismatches")

    # ---- 3. Glynn agrees with Ryser on integers ---------------------------------------
    gmism = 0
    for n in range(1, 9):
        for _ in range(15):
            M = [[rng.randint(-3, 5) for _ in range(n)] for _ in range(n)]
            if permanent_glynn(M) != permanent_ryser(M):
                gmism += 1
    check("Glynn == Ryser on integer matrices", gmism == 0, f"{gmism} mismatches")

    # ---- 4. real-valued matrices: all three agree to floating tolerance ---------------
    def randf():
        return rng.nxt() / 2 ** 32 * 4 - 2
    rmism = 0
    for n in range(1, 7):
        M = [[randf() for _ in range(n)] for _ in range(n)]
        a = permanent_naive(M)
        b = permanent_ryser(M)
        c = permanent_glynn(M)
        if abs(a - b) > 1e-9 * (1 + abs(a)) or abs(a - c) > 1e-8 * (1 + abs(a)):
            rmism += 1
    check("naive/Ryser/Glynn agree on real matrices", rmism == 0, f"{rmism} mismatches")

    # ---- 5. bipartite perfect-matching count vs brute force ---------------------------
    match_bad = 0
    for _ in range(60):
        n = rng.randint(1, 6)
        B = [[rng.randint(0, 1) for _ in range(n)] for _ in range(n)]
        if count_perfect_matchings(B) != brute_count_matchings(B):
            match_bad += 1
    check("perfect matching count == brute force (60 graphs)", match_bad == 0, f"{match_bad} mismatches")

    # complete bipartite K_{n,n} has n! perfect matchings
    for n in range(1, 7):
        K = [[1] * n for _ in range(n)]
        check(f"K_{n},{n} has {n}! perfect matchings", count_perfect_matchings(K) == math.factorial(n))

    # a graph with an isolated left vertex has no perfect matching
    B = [[1, 1, 0], [0, 0, 0], [1, 0, 1]]
    check("no matching when a vertex is isolated", count_perfect_matchings(B) == 0)

    # ---- 6. default permanent() dispatches and validates squareness -------------------
    check("permanent() matches ryser", permanent([[1, 2], [3, 4]]) == permanent_ryser([[1, 2], [3, 4]]))
    try:
        permanent([[1, 2, 3], [4, 5, 6]])
        check("non-square rejected", False)
    except ValueError:
        check("non-square rejected", True)

    # ---- 7. empty matrix -> 1 (empty product) -----------------------------------------
    check("permanent of 0x0 is 1", permanent_ryser([]) == 1)

    # ---- 8. linearity in a row (permanent is multilinear) -----------------------------
    # perm with row i scaled by c equals c * perm
    M = [[rng.randint(1, 5) for _ in range(4)] for _ in range(4)]
    base = permanent_ryser(M)
    M2 = [row[:] for row in M]
    M2[1] = [3 * x for x in M2[1]]
    check("permanent is linear in a row", permanent_ryser(M2) == 3 * base)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
