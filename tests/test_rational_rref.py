"""Tests for rational_rref: exact RREF, rank, null space, and solving over the rationals."""

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rational_rref import (rref, rank, null_space, solve, matvec, is_zero_vector, brute_rank)

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


def random_matrix(rng, r, c, lo=-5, hi=5):
    return [[rng.randint(lo, hi) for _ in range(c)] for _ in range(r)]


def independent(vectors):
    """Are the Fraction vectors linearly independent? (rank equals count.)"""
    if not vectors:
        return True
    return rank([list(v) for v in vectors]) == len(vectors)


# --- known cases ------------------------------------------------------------
check("rank of the singular magic-ish 3x3 is 2", rank([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 2)
check("identity has full rank", rank([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == 3)
check("zero matrix has rank 0", rank([[0, 0], [0, 0]]) == 0)
check("rank of a wide independent matrix", rank([[1, 0, 2], [0, 1, 3]]) == 2)

# RREF of identity is identity
R, piv = rref([[1, 0], [0, 1]])
check("RREF of identity is identity with pivots [0,1]",
      R == [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]] and piv == [0, 1])

# null space of [[1,2,3],[4,5,6],[7,8,9]] is spanned by (1,-2,1)
ns = null_space([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
check("null space is 1-dimensional", len(ns) == 1)
check("null-space vector is proportional to (1,-2,1)",
      ns[0][1] / ns[0][0] == Fraction(-2) and ns[0][2] / ns[0][0] == Fraction(1))

# --- rank matches an independent brute count -------------------------------
rng = LCG(2026)
rank_ok = True
for _ in range(400):
    r = rng.randint(1, 6)
    c = rng.randint(1, 6)
    A = random_matrix(rng, r, c)
    if rank(A) != brute_rank(A):
        rank_ok = False
        print(f"  rank mismatch on {A}")
        break
check("rank matches an independent elimination count (400 random matrices)", rank_ok)

# --- null-space vectors satisfy A x = 0 and are independent ----------------
rng = LCG(4242)
null_ok = True
for _ in range(400):
    r = rng.randint(1, 6)
    c = rng.randint(1, 6)
    A = random_matrix(rng, r, c)
    ns = null_space(A)
    # every basis vector must satisfy A x = 0
    for v in ns:
        if not is_zero_vector(matvec(A, v)):
            null_ok = False
            break
    if not null_ok:
        break
    # dimension: nullity = cols - rank (rank-nullity theorem)
    if len(ns) != c - rank(A):
        null_ok = False
        print(f"  nullity {len(ns)} != cols-rank {c - rank(A)} for {A}")
        break
    # basis vectors are independent
    if not independent(ns):
        null_ok = False
        break
check("null space: A x = 0, dimension = cols - rank, basis independent (400 matrices)", null_ok)

# --- solving: unique, none, infinite ---------------------------------------
check("unique solution recovered", solve([[2, 1], [1, 3]], [3, 5]) == ("unique", [Fraction(4, 5), Fraction(7, 5)]))
check("inconsistent system detected", solve([[1, 1], [1, 1]], [1, 2])[0] == "none")
kind, data = solve([[1, 1, 1]], [6])
check("underdetermined system reported as infinite", kind == "infinite")

# --- solve() results actually satisfy the system --------------------------
rng = LCG(777)
solve_ok = True
for _ in range(400):
    n = rng.randint(1, 5)
    m = rng.randint(1, 5)
    A = random_matrix(rng, m, n)
    # build a consistent b from a random x with some probability, else random b
    if rng.rand() % 2 == 0:
        x_true = [Fraction(rng.randint(-5, 5)) for _ in range(n)]
        b = [sum(Fraction(A[i][j]) * x_true[j] for j in range(n)) for i in range(m)]
    else:
        b = [Fraction(rng.randint(-10, 10)) for _ in range(m)]
    kind, data = solve(A, b)
    if kind == "unique":
        if matvec(A, data) != [Fraction(x) for x in b]:
            solve_ok = False
            break
    elif kind == "infinite":
        x0, basis = data
        # particular solution works
        if matvec(A, x0) != [Fraction(x) for x in b]:
            solve_ok = False
            break
        # x0 + any null-space combo still solves it
        for v in basis:
            xv = [x0[i] + 3 * v[i] for i in range(n)]
            if matvec(A, xv) != [Fraction(x) for x in b]:
                solve_ok = False
                break
        if not solve_ok:
            break
    # kind == "none": nothing to satisfy
check("solve() results satisfy A x = b (unique + infinite families) (400 systems)", solve_ok)

# --- consistent systems from a known x are always solvable -----------------
rng = LCG(555)
consistent_ok = True
for _ in range(300):
    n = rng.randint(1, 5)
    m = rng.randint(1, 5)
    A = random_matrix(rng, m, n)
    x_true = [Fraction(rng.randint(-6, 6)) for _ in range(n)]
    b = [sum(Fraction(A[i][j]) * x_true[j] for j in range(n)) for i in range(m)]
    kind, data = solve(A, b)
    if kind == "none":
        consistent_ok = False
        break
check("a system built from a real solution is never reported inconsistent (300)", consistent_ok)

# --- fractional input works exactly ----------------------------------------
A = [[Fraction(1, 2), Fraction(1, 3)], [Fraction(1, 4), Fraction(1, 5)]]
b = [Fraction(1), Fraction(1)]
kind, x = solve(A, b)
check("fractional input solved exactly", kind == "unique" and matvec(A, x) == [Fraction(1), Fraction(1)])

# --- RREF is idempotent: RREF(RREF(A)) == RREF(A) --------------------------
rng = LCG(31337)
idem_ok = True
for _ in range(100):
    A = random_matrix(rng, rng.randint(1, 5), rng.randint(1, 5))
    R1, _ = rref(A)
    R2, _ = rref(R1)
    if R1 != R2:
        idem_ok = False
        break
check("RREF is idempotent", idem_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all rational_rref tests passed")
