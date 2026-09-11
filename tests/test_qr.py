"""Tests for qr.py -- QR decomposition and least squares.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. The factorization is checked by
Q^T Q = I and Q R = A; least squares by the normal-equation orthogonality of the residual.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import qr as QR  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def mclose(A, B, tol=1e-9):
    return all(approx(A[i][j], B[i][j], tol) for i in range(len(A)) for j in range(len(A[0])))


def _raises(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


# --- square QR --------------------------------------------------------------
A = [[12, -51, 4], [6, 167, -68], [-4, 24, -41]]
Q, R = QR.qr_decompose(A)
check("Q has orthonormal columns (Q^T Q = I)", QR.is_orthonormal(Q))
check("Q R reconstructs A", mclose(QR.reconstruct(Q, R), A))
check("R is upper triangular", all(approx(R[i][j], 0.0) for i in range(3) for j in range(i)))
check("R has positive diagonal (this MGS convention)", all(R[i][i] > 0 for i in range(3)))

# --- solving a square system ------------------------------------------------
x = QR.solve([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], [8, -11, -3])
check("QR solves a square system", all(approx(a, b) for a, b in zip(x, [2, 3, -1])))
check("solve rejects a rectangular matrix (use lstsq)",
      _raises(lambda: QR.solve([[1, 2], [3, 4], [5, 6]], [1, 2, 3])))

# --- least squares on an exact linear fit ----------------------------------
pts = [(0, 1), (1, 3), (2, 5), (3, 7), (4, 9)]     # exactly y = 2x + 1
Am = [[px, 1] for px, _ in pts]
bm = [py for _, py in pts]
sol = QR.lstsq(Am, bm)
check("least squares recovers the exact line", approx(sol[0], 2.0, 1e-9) and approx(sol[1], 1.0, 1e-9))
check("exact-fit residual is ~0", QR.residual_norm(Am, sol, bm) < 1e-9)

# --- least squares minimizes the residual (orthogonality) ------------------
def lcg(seed):
    s = seed
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield s >> 8


gen = lcg(1)


def rf(a, b):
    return a + (b - a) * (next(gen) / (1 << 24))


noisy = [(px, 2 * px + 1 + rf(-0.5, 0.5)) for px in range(30)]
An = [[px, 1] for px, _ in noisy]
bn = [py for _, py in noisy]
xn = QR.lstsq(An, bn)
r = QR.residual(An, xn, bn)
c0 = [An[i][0] for i in range(30)]
c1 = [An[i][1] for i in range(30)]
check("least-squares residual is orthogonal to column 0", abs(sum(r[i] * c0[i] for i in range(30))) < 1e-8)
check("least-squares residual is orthogonal to column 1", abs(sum(r[i] * c1[i] for i in range(30))) < 1e-8)
check("fitted slope is close to the true 2", approx(xn[0], 2.0, 0.1))
# the LS solution beats a perturbed one (residual is minimal)
worse = [xn[0] + 0.5, xn[1]]
check("least-squares residual is smaller than a perturbed solution's",
      QR.residual_norm(An, xn, bn) < QR.residual_norm(An, worse, bn))

# --- quadratic least-squares fit -------------------------------------------
qpts = [(px, 3 * px * px - 2 * px + 5) for px in range(8)]     # exact quadratic
Aq = [[px * px, px, 1] for px, _ in qpts]
bq = [py for _, py in qpts]
sq = QR.lstsq(Aq, bq)
check("least squares fits a quadratic exactly", all(approx(a, b, 1e-6) for a, b in zip(sq, [3, -2, 5])))

# --- edge cases -------------------------------------------------------------
check("1x1 QR", mclose(QR.reconstruct(*QR.qr_decompose([[5]])), [[5]]))
try:
    QR.qr_decompose([[1, 2], [2, 4]])       # linearly dependent columns
    check("rejects a rank-deficient matrix", False)
except ValueError:
    check("rejects a rank-deficient matrix", True)
try:
    QR.qr_decompose([[1, 2, 3]])            # m < n
    check("rejects m < n", False)
except ValueError:
    check("rejects m < n", True)

# --- exhaustive: random square systems solve accurately --------------------
bad = 0
tested = 0
for _ in range(300):
    n = 1 + next(gen) % 5
    M = [[rf(-6, 6) for _ in range(n)] for _ in range(n)]
    rhs = [rf(-6, 6) for _ in range(n)]
    try:
        xs = QR.solve(M, rhs)
    except ValueError:
        continue
    tested += 1
    if QR.residual_norm(M, xs, rhs) > 1e-6:
        bad += 1
check(f"all {tested} random square QR solves have a tiny residual", bad == 0)

# --- Q orthonormality on random tall matrices ------------------------------
ortho_ok = True
for _ in range(100):
    m = 3 + next(gen) % 6
    n = 1 + next(gen) % m
    M = [[rf(-5, 5) for _ in range(n)] for _ in range(m)]
    try:
        Qr, Rr = QR.qr_decompose(M)
    except ValueError:
        continue
    if not QR.is_orthonormal(Qr, 1e-8) or not mclose(QR.reconstruct(Qr, Rr), M, 1e-7):
        ortho_ok = False
        break
check("Q is orthonormal and QR=A on random tall matrices", ortho_ok)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall qr tests passed")
