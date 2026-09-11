"""Tests for eigen.py -- power iteration and eigenvalue methods.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Eigenpairs are checked by
A v = lambda v, the trace identity, and small analytic cases.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import eigen as E  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol=1e-5):
    return abs(a - b) <= tol


# --- power iteration: dominant eigenvalue ----------------------------------
A = [[2, 1], [1, 2]]        # eigenvalues 3 (dominant) and 1
lam, v, it = E.power_iteration(A)
check("power iteration finds the dominant eigenvalue 3", approx(lam, 3.0))
check("the eigenpair satisfies A v = lambda v", E.residual_norm(A, lam, v) < 1e-4)
check("the eigenvector is normalized", approx(sum(x * x for x in v), 1.0, 1e-6))
# a matrix with a clearly dominant eigenvalue
B = [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
lamB, vB, _ = E.power_iteration(B)
check("dominant eigenvalue of a 3x3 satisfies the equation", E.residual_norm(B, lamB, vB) < 1e-4)
check("dominant eigenvalue is the largest", lamB > 4.0)

# --- Rayleigh quotient ------------------------------------------------------
check("Rayleigh quotient of an eigenvector gives its eigenvalue",
      approx(E.rayleigh_quotient([[3, 0], [0, 1]], [1, 0]), 3.0))
check("Rayleigh quotient of the other eigenvector", approx(E.rayleigh_quotient([[3, 0], [0, 1]], [0, 1]), 1.0))

# --- characteristic polynomial cross-check (2x2) ---------------------------
check("dominant eigenvalue matches the characteristic polynomial",
      approx(E.power_iteration(A)[0], E.characteristic_2x2(A)[0]))
check("characteristic_2x2 of [[2,1],[1,2]] is [3,1]",
      all(approx(a, b) for a, b in zip(E.characteristic_2x2(A), [3, 1])))
check("characteristic_2x2 detects a complex pair",
      isinstance(E.characteristic_2x2([[0, -1], [1, 0]])[0], complex))     # rotation: eigenvalues +-i

# --- inverse iteration: eigenvalue nearest a shift -------------------------
check("inverse iteration finds the eigenvalue nearest 0.5 (=1)", approx(E.inverse_iteration(A, 0.5)[0], 1.0))
check("inverse iteration nearest 2.8 finds 3", approx(E.inverse_iteration(A, 2.8)[0], 3.0))
lam_small, v_small, _ = E.inverse_iteration(B, 1.5)
check("inverse iteration eigenpair satisfies A v = lambda v", E.residual_norm(B, lam_small, v_small) < 1e-4)

# --- all eigenvalues of a symmetric matrix by deflation --------------------
vals, vecs = E.eigenvalues_symmetric(A)
check("deflation recovers both eigenvalues 3 and 1", approx(vals[0], 3.0) and approx(vals[1], 1.0))
check("all deflated eigenpairs satisfy A v = lambda v",
      all(E.residual_norm(A, l, vv) < 1e-3 for l, vv in zip(vals, vecs)))
# a diagonal matrix: eigenvalues are the diagonal
D = [[5, 0, 0], [0, 3, 0], [0, 0, 1]]
dvals, _ = E.eigenvalues_symmetric(D)
check("diagonal matrix eigenvalues are its diagonal", all(approx(a, b) for a, b in zip(sorted(dvals, reverse=True), [5, 3, 1])))

# --- trace and determinant identities --------------------------------------
check("sum of eigenvalues equals the trace (2x2)", approx(sum(vals), E.trace(A)))
prod = 1.0
for lv in vals:
    prod *= lv
det_A = A[0][0] * A[1][1] - A[0][1] * A[1][0]
check("product of eigenvalues equals the determinant", approx(prod, det_A, 1e-4))

# --- larger random symmetric matrices --------------------------------------
def lcg(seed):
    s = seed
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield s >> 8


gen = lcg(1)


def rf(a, b):
    return a + (b - a) * (next(gen) / (1 << 24))


trace_ok = resid_ok = True
for _ in range(30):
    n = 2 + next(gen) % 4
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            M[i][j] = M[j][i] = rf(-4, 4)
    vv, ve = E.eigenvalues_symmetric(M)
    if not approx(sum(vv), E.trace(M), 1e-4 * (1 + abs(E.trace(M)))):
        trace_ok = False
    if max(E.residual_norm(M, l, v) for l, v in zip(vv, ve)) > 1e-3:
        resid_ok = False
check("sum of eigenvalues = trace on 30 random symmetric matrices", trace_ok)
check("every eigenpair satisfies A v = lambda v on random matrices", resid_ok)

# --- a negative dominant eigenvalue doesn't flip the sign ------------------
neg = [[-5, 0], [0, 2]]      # dominant eigenvalue -5
lamn, vn, _ = E.power_iteration(neg)
check("handles a negative dominant eigenvalue", approx(lamn, -5.0))
check("negative-eigenvalue eigenpair is valid", E.residual_norm(neg, lamn, vn) < 1e-4)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall eigen tests passed")
