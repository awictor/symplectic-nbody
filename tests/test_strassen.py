"""Tests for strassen: sub-cubic matrix multiplication vs the schoolbook product."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import strassen
from strassen import multiply, schoolbook, identity, equal

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


def random_matrix(rng, r, c, lo=-9, hi=9):
    return [[rng.randint(lo, hi) for _ in range(c)] for _ in range(r)]


# --- known products ---------------------------------------------------------
a = [[1, 2], [3, 4]]
b = [[5, 6], [7, 8]]
check("2x2 known product", multiply(a, b) == [[19, 22], [43, 50]])
check("identity multiply is a no-op", equal(multiply(a, identity(2)), a))
check("multiply by identity on the left", equal(multiply(identity(2), b), b))
check("1x1 product", multiply([[3]], [[7]]) == [[21]])

# --- force deep recursion (small cutoff) and check vs schoolbook -----------
orig_cutoff = strassen._CUTOFF
strassen._CUTOFF = 2
try:
    rng = LCG(2026)
    square_ok = True
    for _ in range(200):
        n = rng.randint(1, 40)
        a = random_matrix(rng, n, n)
        b = random_matrix(rng, n, n)
        if not equal(multiply(a, b), schoolbook(a, b)):
            square_ok = False
            print(f"  square mismatch at n={n}")
            break
    check("Strassen (deep recursion) matches schoolbook on square matrices (200)", square_ok)

    # rectangular / mismatched shapes
    rect_ok = True
    for _ in range(200):
        r = rng.randint(1, 30)
        k = rng.randint(1, 30)
        c = rng.randint(1, 30)
        a = random_matrix(rng, r, k)
        b = random_matrix(rng, k, c)
        if not equal(multiply(a, b), schoolbook(a, b)):
            rect_ok = False
            print(f"  rect mismatch at {r}x{k} * {k}x{c}")
            break
    check("Strassen matches schoolbook on rectangular matrices (200)", rect_ok)
finally:
    strassen._CUTOFF = orig_cutoff

# --- larger matrices at the real cutoff ------------------------------------
rng = LCG(4242)
big_ok = True
for n in (33, 50, 64, 100):
    a = random_matrix(rng, n, n)
    b = random_matrix(rng, n, n)
    if not equal(multiply(a, b), schoolbook(a, b)):
        big_ok = False
        print(f"  big mismatch at n={n}")
        break
check("Strassen matches schoolbook on 33/50/64/100-size matrices", big_ok)

# --- float matrices ---------------------------------------------------------
strassen._CUTOFF = 2
try:
    rng = LCG(777)
    float_ok = True
    for _ in range(100):
        n = rng.randint(2, 20)
        a = [[rng.randint(-100, 100) / 10.0 for _ in range(n)] for _ in range(n)]
        b = [[rng.randint(-100, 100) / 10.0 for _ in range(n)] for _ in range(n)]
        if not equal(multiply(a, b), schoolbook(a, b), tol=1e-6):
            float_ok = False
            break
    check("Strassen matches schoolbook on float matrices", float_ok)
finally:
    strassen._CUTOFF = orig_cutoff

# --- associativity: (AB)C == A(BC) -----------------------------------------
strassen._CUTOFF = 2
try:
    rng = LCG(555)
    assoc_ok = True
    for _ in range(100):
        n = rng.randint(2, 16)
        a = random_matrix(rng, n, n, -5, 5)
        b = random_matrix(rng, n, n, -5, 5)
        c = random_matrix(rng, n, n, -5, 5)
        ab_c = multiply(multiply(a, b), c)
        a_bc = multiply(a, multiply(b, c))
        if not equal(ab_c, a_bc):
            assoc_ok = False
            break
    check("Strassen respects associativity (AB)C == A(BC)", assoc_ok)
finally:
    strassen._CUTOFF = orig_cutoff

# --- dimension mismatch is rejected ----------------------------------------
try:
    multiply([[1, 2, 3]], [[1, 2]])       # 1x3 * 1x2 -> mismatch
    raised = False
except ValueError:
    raised = True
check("a dimension mismatch raises ValueError", raised)

# --- zero and single-row/column shapes -------------------------------------
check("row times column gives a 1x1 dot product",
      multiply([[1, 2, 3]], [[4], [5], [6]]) == [[32]])
check("column times row gives an outer product",
      multiply([[1], [2]], [[3, 4]]) == [[3, 4], [6, 8]])

# --- exact integer results for large integer matrices ----------------------
strassen._CUTOFF = 4
try:
    rng = LCG(31337)
    n = 20
    a = [[rng.randint(1, 10 ** 6) for _ in range(n)] for _ in range(n)]
    b = [[rng.randint(1, 10 ** 6) for _ in range(n)] for _ in range(n)]
    check("Strassen is exact on large-integer matrices",
          multiply(a, b) == schoolbook(a, b))
finally:
    strassen._CUTOFF = orig_cutoff

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all strassen tests passed")
