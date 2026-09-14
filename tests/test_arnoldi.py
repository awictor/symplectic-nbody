"""Validate Arnoldi: the factorization identity, orthonormality, and Ritz values vs full eigensolvers."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import arnoldi
import qr_algorithm
import lanczos


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24) * 2 - 1


def rand_matrix(n, seed):
    r = _R(seed)
    return [[r.u() for _ in range(n)] for _ in range(n)]


def symmetrize(A):
    n = len(A)
    return [[(A[i][j] + A[j][i]) / 2 for j in range(n)] for i in range(n)]


def main():
    print("Arnoldi tests")

    n = 12
    A = rand_matrix(n, 42)
    mv = arnoldi.matvec_from_matrix(A)

    # --- Arnoldi factorization identity A Q_m = Q_{m+1} H ---
    Q, H, brk = arnoldi.arnoldi_factorization(mv, n, m=8, seed=7)
    res = arnoldi.factorization_residual(mv, Q, H)
    check("A Q_m = Q_{m+1} H to machine precision", res < 1e-9)

    # --- Q has orthonormal columns ---
    ortho = 0.0
    for i in range(len(Q)):
        for j in range(len(Q)):
            d = sum(Q[i][k] * Q[j][k] for k in range(n))
            target = 1.0 if i == j else 0.0
            ortho = max(ortho, abs(d - target))
    check("Q columns orthonormal", ortho < 1e-9)

    # --- H is upper Hessenberg (zero below the first subdiagonal) ---
    k = len(H[0])
    below = 0.0
    for j in range(k):
        for i in range(j + 2, len(H)):
            below = max(below, abs(H[i][j]))
    check("H upper Hessenberg", below < 1e-12)

    # --- Ritz values match the dominant true eigenvalues (non-symmetric) ---
    true = qr_algorithm.eigenvalues(A)
    true.sort(key=lambda z: -abs(z))
    # full Krylov space (m = n) must reproduce the ENTIRE spectrum
    ritz_full = arnoldi.ritz_values(mv, n, m=n, seed=3)
    ok = True
    for i in range(n):
        if abs(ritz_full[i] - true[i]) > 1e-5:
            ok = False
    check("full Arnoldi (m=n) recovers whole spectrum", ok)

    # --- dominant eigenvalue converges with small m (well-separated real dominant) ---
    # Random matrices often have a complex conjugate pair on top, which Arnoldi resolves
    # slowly; a clean spectral gap (one boosted diagonal) makes the dominant mode converge
    # in far fewer steps, the regime Arnoldi is actually used for.
    Asep = rand_matrix(n, 42)
    Asep[0][0] += 15.0
    mvSep = arnoldi.matvec_from_matrix(Asep)
    tsep = qr_algorithm.eigenvalues(Asep)
    tsep.sort(key=lambda z: -abs(z))
    ritz_small = arnoldi.ritz_values(mvSep, n, m=6, seed=3)
    check("dominant Ritz value near true dominant (m=6, gapped)",
          abs(ritz_small[0] - tsep[0]) < 1e-3)

    # --- symmetric case: Ritz values match Lanczos and true eigenvalues ---
    S = symmetrize(rand_matrix(n, 99))
    mvS = arnoldi.matvec_from_matrix(S)
    trueS = qr_algorithm.eigenvalues(S)
    trueS = sorted([z.real for z in trueS], reverse=True)
    ritzS = arnoldi.ritz_values(mvS, n, m=n, seed=5)
    ritzS = sorted([z.real for z in ritzS], reverse=True)
    okS = all(abs(ritzS[i] - trueS[i]) < 1e-5 for i in range(n))
    check("symmetric Arnoldi recovers real spectrum", okS)
    # imaginary parts vanish for a symmetric matrix
    maxim = max(abs(z.imag) for z in arnoldi.ritz_values(mvS, n, m=n, seed=5))
    check("symmetric spectrum is real (tiny imag)", maxim < 1e-6)

    # --- Ritz vector is an approximate eigenvector: ||A x - lam x|| small ---
    pairs = arnoldi.ritz_pairs(mvS, n, m=n, seed=5)
    lam, x = pairs[0]
    Ax = mvS(x)
    r = math.sqrt(sum((Ax[i] - lam * x[i]) ** 2 for i in range(n)))
    check("dominant Ritz vector residual small", r < 1e-4)

    # --- happy breakdown on a rank-deficient Krylov space ---
    # A diagonal matrix started from a vector spanning only 3 eigendirections:
    # the Krylov space closes after 3 steps -> breakdown.
    D = [[0.0] * 6 for _ in range(6)]
    for i in range(6):
        D[i][i] = float(i + 1)
    mvD = arnoldi.matvec_from_matrix(D)
    v0 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    Qd, Hd, brkd = arnoldi.arnoldi_factorization(mvD, 6, m=6, v0=v0)
    check("happy breakdown detected at step 3", brkd == 3)
    # the 3 Ritz values are exactly the 3 active eigenvalues {1,2,3}
    rd = arnoldi.ritz_values(mvD, 6, m=6, v0=v0)
    rd_real = sorted(z.real for z in rd)
    check("breakdown Ritz values are the active eigenvalues",
          len(rd_real) == 3 and all(abs(rd_real[i] - (i + 1)) < 1e-9 for i in range(3)))

    # --- reproducible per seed, differs across seeds (random start vector) ---
    r1 = arnoldi.ritz_values(mv, n, m=5, seed=100)
    r2 = arnoldi.ritz_values(mv, n, m=5, seed=100)
    r3 = arnoldi.ritz_values(mv, n, m=5, seed=200)
    check("reproducible per seed", all(r1[i] == r2[i] for i in range(len(r1))))
    check("differs across seeds", any(abs(r1[i] - r3[i]) > 1e-9 for i in range(len(r1))))

    # --- dominant Ritz value is seed-robust (converges regardless of start) ---
    # Use the gapped matrix: with a clear spectral gap the dominant Ritz value is
    # start-independent by m=6; on a matrix with a top complex pair it isn't yet.
    g1 = arnoldi.ritz_values(mvSep, n, m=6, seed=100)
    g3 = arnoldi.ritz_values(mvSep, n, m=6, seed=200)
    check("dominant Ritz value seed-robust (gapped)", abs(g1[0] - g3[0]) < 1e-2)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
