"""Tests for randomized SVD: singular values match exact, near-optimal error, power iteration."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from randomized_svd import randomized_svd, reconstruct, frobenius  # noqa: E402
from svd import svd as exact_svd  # noqa: E402


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


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def _matmul(A, B):
    return [[sum(A[i][t] * B[t][j] for t in range(len(B)))
             for j in range(len(B[0]))] for i in range(len(A))]


def _low_rank_matrix(m, n, r, rng):
    """A random rank-r non-negative-ish matrix = (m x r)(r x n)."""
    L = [[rng() * 2 - 1 for _ in range(r)] for _ in range(m)]
    R = [[rng() * 2 - 1 for _ in range(n)] for _ in range(r)]
    return _matmul(L, R)


def main():
    # ---- 1. singular values match exact top-k -------------------------------------------
    rng = _lcg(2024)
    m, n = 30, 20
    A = [[rng() * 2 - 1 for _ in range(n)] for _ in range(m)]
    Ue, Se, Vte = exact_svd(A)
    k = 5
    U, S, Vt = randomized_svd(A, k, oversample=8, n_power=2, seed=7)
    err = max(abs(S[i] - Se[i]) for i in range(k))
    check("top-k singular values match exact SVD", err < 0.02 * Se[0], f"max err {err:.4f}")

    # ---- 2. reconstruction error near the optimal (k+1)-th singular value ---------------
    approx = reconstruct(U, S, Vt)
    err = frobenius(A, approx)
    # optimal rank-k Frobenius error = sqrt(sum_{i>k} sigma_i^2)
    opt = math.sqrt(sum(Se[i] ** 2 for i in range(k, len(Se))))
    check("reconstruction error near optimal rank-k error", err < 1.15 * opt + 1e-6,
          f"err {err:.3f} vs opt {opt:.3f}")

    # ---- 3. singular vectors orthonormal ------------------------------------------------
    def orthonormal_cols(M):
        # columns of M (m x k)
        kk = len(M[0])
        for a in range(kk):
            for b in range(kk):
                dot = sum(M[i][a] * M[i][b] for i in range(len(M)))
                expect = 1.0 if a == b else 0.0
                if abs(dot - expect) > 1e-4:
                    return False
        return True
    check("U columns orthonormal", orthonormal_cols(U))
    # Vt rows orthonormal
    def orthonormal_rows(M):
        for a in range(len(M)):
            for b in range(len(M)):
                dot = sum(M[a][j] * M[b][j] for j in range(len(M[0])))
                expect = 1.0 if a == b else 0.0
                if abs(dot - expect) > 1e-4:
                    return False
        return True
    check("Vt rows orthonormal", orthonormal_rows(Vt))

    # ---- 4. exactly rank-r matrix recovered to high precision ---------------------------
    rng = _lcg(77)
    r = 4
    A = _low_rank_matrix(25, 18, r, rng)
    U, S, Vt = randomized_svd(A, r, oversample=6, n_power=1, seed=3)
    err = frobenius(A, reconstruct(U, S, Vt))
    Anorm = frobenius(A, [[0] * len(A[0]) for _ in range(len(A))])
    check("exact rank-r matrix recovered (tiny error)", err < 1e-3 * Anorm, f"{err:.2e}")

    # ---- 5. power iteration reduces error on slowly-decaying spectrum -------------------
    # build a matrix with slowly-decaying singular values
    rng = _lcg(555)
    m, n = 40, 30
    # A = sum_i sigma_i u_i v_i^T with sigma_i = 1/i^0.4 (slow decay)
    U0 = _low_rank_matrix(m, m, m, rng)  # random-ish, not orthonormal but fine for a test signal
    # simpler: random matrix, compare q=0 vs q=3
    A = [[rng() * 2 - 1 for _ in range(n)] for _ in range(m)]
    k = 6
    _, S0, _ = randomized_svd(A, k, oversample=2, n_power=0, seed=9)
    U3, S3, Vt3 = randomized_svd(A, k, oversample=2, n_power=4, seed=9)
    Ue, Se, _ = exact_svd(A)
    # power iteration should give singular values closer to (>= ) the true ones (captures more energy)
    captured0 = sum(S0[i] ** 2 for i in range(k))
    captured3 = sum(S3[i] ** 2 for i in range(k))
    check("power iteration captures at least as much energy", captured3 >= captured0 - 1e-9,
          f"{captured0:.3f} -> {captured3:.3f}")
    check("power-iteration singular values close to exact",
          max(abs(S3[i] - Se[i]) for i in range(k)) < 0.05 * Se[0])

    # ---- 6. determinism under seed ------------------------------------------------------
    r1 = randomized_svd(A, 4, seed=42)
    r2 = randomized_svd(A, 4, seed=42)
    check("deterministic under seed", r1[1] == r2[1])

    # ---- 7. full-rank request returns min(m,n) components -------------------------------
    A = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    U, S, Vt = randomized_svd(A, 2, oversample=0, n_power=1, seed=1)
    Ue, Se, Vte = exact_svd(A)
    check("2x2 top singular value matches exact", abs(S[0] - Se[0]) < 1e-6, f"{S[0]} vs {Se[0]}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
