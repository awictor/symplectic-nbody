"""Tests for linsolve.py -- Gaussian elimination and LU decomposition.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Solutions are checked by
substitution and residual norm; the factorization by P A = L U.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import linsolve as LA  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def vclose(u, v, tol=1e-9):
    return all(abs(a - b) <= tol for a, b in zip(u, v))


# --- solving a known system ------------------------------------------------
A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
b = [8, -11, -3]
x = LA.solve(A, b)
check("solves a 3x3 system", vclose(x, [2, 3, -1]))
check("residual is ~0", LA.residual_norm(A, x, b) < 1e-9)
check("solves a 1x1 system", vclose(LA.solve([[5]], [10]), [2.0]))
check("solves a diagonal system", vclose(LA.solve([[2, 0], [0, 4]], [6, 8]), [3, 2]))
try:
    LA.solve([[1, 2]], [1])
    check("rejects a non-square matrix", False)
except ValueError:
    check("rejects a non-square matrix", True)
try:
    LA.solve([[1, 2], [2, 4]], [1, 2])   # singular
    check("rejects a singular matrix", False)
except ValueError:
    check("rejects a singular matrix", True)

# --- partial pivoting handles a zero pivot ---------------------------------
check("pivoting solves a system with a zero on the diagonal",
      vclose(LA.solve([[0, 1], [1, 0]], [1, 2]), [2, 1]))
# a matrix that needs pivoting for stability (tiny pivot)
tiny = [[1e-15, 1], [1, 1]]
xt = LA.solve(tiny, [1, 2])
check("pivoting keeps a tiny-pivot system accurate", LA.residual_norm(tiny, xt, [1, 2]) < 1e-6)

# --- LU factorization: P A = L U -------------------------------------------
L, U, perm, sign = LA.lu_decompose(A)
PA = [A[perm[i]] for i in range(len(A))]
LU = LA.matmul(L, U)
check("P A = L U", all(approx(PA[i][j], LU[i][j]) for i in range(3) for j in range(3)))
check("L is unit lower triangular", all(L[i][i] == 1.0 for i in range(3)) and all(L[i][j] == 0 for i in range(3) for j in range(i + 1, 3)))
check("U is upper triangular", all(U[i][j] == 0 for i in range(3) for j in range(i)))
check("sign is +-1", sign in (1, -1))

# --- reusing a factorization for multiple right-hand sides -----------------
for rhs in ([8, -11, -3], [1, 0, 0], [3, 3, 3]):
    xs = LA.solve_lu(L, U, perm, rhs)
    check(f"solve_lu reuses the factorization for rhs {rhs}", LA.residual_norm(A, xs, rhs) < 1e-9)

# --- determinant ------------------------------------------------------------
check("det of a 2x2", approx(LA.determinant([[1, 2], [3, 4]]), -2.0))
check("det of the identity is 1", approx(LA.determinant(LA.identity(4)), 1.0))
check("det of a triangular matrix is the diagonal product",
      approx(LA.determinant([[2, 5, 1], [0, 3, 4], [0, 0, 7]]), 42.0))
check("det of a singular matrix is 0", LA.determinant([[1, 2], [2, 4]]) == 0.0)
check("det matches the known 3x3", approx(LA.determinant(A), -1.0, 1e-9))
# det(AB) = det(A) det(B)
B = [[1, 2, 0], [0, 1, 3], [2, 0, 1]]
check("det is multiplicative", approx(LA.determinant(LA.matmul(A, B)),
                                      LA.determinant(A) * LA.determinant(B), 1e-6))

# --- inverse ----------------------------------------------------------------
inv = LA.inverse(A)
prod = LA.matmul(A, inv)
check("A * inverse = identity", all(approx(prod[i][j], 1.0 if i == j else 0.0) for i in range(3) for j in range(3)))
prod2 = LA.matmul(inv, A)
check("inverse * A = identity too", all(approx(prod2[i][j], 1.0 if i == j else 0.0) for i in range(3) for j in range(3)))
check("inverse of the identity is the identity",
      all(approx(LA.inverse(LA.identity(3))[i][j], 1.0 if i == j else 0.0) for i in range(3) for j in range(3)))

# --- matvec / matmul helpers -----------------------------------------------
check("matvec computes A x", vclose(LA.matvec([[1, 2], [3, 4]], [1, 1]), [3, 7]))
check("matmul computes A B", LA.matmul([[1, 0], [0, 1]], [[5, 6], [7, 8]]) == [[5, 6], [7, 8]])

# --- exhaustive random check: solve then verify residual -------------------
def lcg(seed):
    s = seed
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield s >> 8


gen = lcg(1)


def rf(a, b):
    return a + (b - a) * (next(gen) / (1 << 24))


bad = 0
tested = 0
for _ in range(400):
    n = 1 + next(gen) % 6
    M = [[rf(-8, 8) for _ in range(n)] for _ in range(n)]
    rhs = [rf(-8, 8) for _ in range(n)]
    try:
        xs = LA.solve(M, rhs)
    except ValueError:
        continue
    tested += 1
    if LA.residual_norm(M, xs, rhs) > 1e-6:
        bad += 1
check(f"all {tested} random solves have a tiny residual", bad == 0)

# --- determinant sign flips with a row swap --------------------------------
M = [[1, 2], [3, 4]]
Mswap = [[3, 4], [1, 2]]
check("swapping two rows negates the determinant",
      approx(LA.determinant(M), -LA.determinant(Mswap)))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall linsolve tests passed")
