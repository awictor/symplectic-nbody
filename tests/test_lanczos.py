"""Tests for lanczos: Ritz values match dense eigensolver, Ritz vectors valid, matrix-free Laplacian."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lanczos import (lanczos_tridiagonal, tridiagonal_eigen, eigenvalues, matvec_from_matrix,  # noqa: E402
                     largest_eigenvalues, smallest_eigenvalues, ritz_pairs, _dot, _norm)
import jacobi_eigen  # noqa: E402


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

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24) * 2 - 1


def random_symmetric(n, rng):
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            v = rng.u() * 5
            A[i][j] = v
            A[j][i] = v
    return A


def main():
    rng = LCG(2024)

    # ---- 1. full-iteration Lanczos recovers the whole spectrum vs dense Jacobi --------
    for n in (5, 8, 12, 20):
        A = random_symmetric(n, rng)
        lz = eigenvalues(matvec_from_matrix(A), n)          # m = n -> full spectrum
        jv = sorted(jacobi_eigen.sorted_eigen(A)[0])
        maxdiff = max(abs(a - b) for a, b in zip(lz, jv))
        check(f"full Lanczos matches dense spectrum (n={n})", maxdiff < 1e-8,
              f"max diff {maxdiff:.2e}")

    # ---- 2. extreme eigenvalues converge in far fewer than n steps --------------------
    n = 60
    A = random_symmetric(n, rng)
    jv = sorted(jacobi_eigen.sorted_eigen(A)[0])
    true_max = jv[-1]
    true_min = jv[0]
    got_max = largest_eigenvalues(matvec_from_matrix(A), n, k=1, m=30)[0]
    got_min = smallest_eigenvalues(matvec_from_matrix(A), n, k=1, m=30)[0]
    check("largest eigenvalue converges in m<<n steps", abs(got_max - true_max) < 1e-6,
          f"{got_max:.6f} vs {true_max:.6f}")
    check("smallest eigenvalue converges in m<<n steps", abs(got_min - true_min) < 1e-6,
          f"{got_min:.6f} vs {true_min:.6f}")

    # top-3 largest
    got3 = largest_eigenvalues(matvec_from_matrix(A), n, k=3, m=40)
    true3 = jv[-3:][::-1]
    check("top-3 largest match", all(abs(a - b) < 1e-5 for a, b in zip(got3, true3)),
          f"{[round(x,4) for x in got3]} vs {[round(x,4) for x in true3]}")

    # ---- 3. Ritz vectors satisfy the eigen-relation -----------------------------------
    n = 15
    A = random_symmetric(n, rng)
    mv = matvec_from_matrix(A)
    vals, vecs = ritz_pairs(mv, n)
    worst_res = 0.0
    for lam, v in zip(vals, vecs):
        Av = mv(v)
        res = math.sqrt(sum((Av[i] - lam * v[i]) ** 2 for i in range(n)))
        worst_res = max(worst_res, res)
    check("Ritz vectors satisfy A v = lambda v", worst_res < 1e-7, f"worst residual {worst_res:.2e}")
    # Ritz vectors are (near) unit norm
    check("Ritz vectors are unit norm", all(abs(_norm(v) - 1.0) < 1e-6 for v in vecs))

    # ---- 4. Lanczos basis is orthonormal and T is symmetric tridiagonal ---------------
    n = 12
    A = random_symmetric(n, rng)
    alpha, beta, basis = lanczos_tridiagonal(matvec_from_matrix(A), n)
    ortho = True
    for i in range(len(basis)):
        for j in range(len(basis)):
            ip = _dot(basis[i], basis[j])
            if abs(ip - (1.0 if i == j else 0.0)) > 1e-8:
                ortho = False
    check("Lanczos basis is orthonormal (reorthogonalisation)", ortho)
    check("alpha/beta lengths consistent with tridiagonal", len(beta) == len(alpha) - 1)

    # ---- 5. tridiagonal eigensolver on a hand matrix ---------------------------------
    # symmetric tridiagonal [[2,-1,0],[-1,2,-1],[0,-1,2]] has eigenvalues 2, 2+-sqrt2
    vals, _ = tridiagonal_eigen([2, 2, 2], [-1, -1])
    expected = sorted([2 - math.sqrt(2), 2, 2 + math.sqrt(2)])
    check("tridiagonal eigenvalues correct", all(abs(a - b) < 1e-10 for a, b in zip(vals, expected)),
          f"{[round(v,5) for v in vals]}")

    # ---- 6. matrix-free: graph Laplacian, smallest eigenvalue is exactly 0 ------------
    # a path graph on 10 nodes; L = D - A, matrix-free
    n = 10
    edges = [(i, i + 1) for i in range(n - 1)]
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    def laplacian_mv(x):
        return [len(adj[i]) * x[i] - sum(x[j] for j in adj[i]) for i in range(n)]

    small = smallest_eigenvalues(laplacian_mv, n, k=1, m=n)[0]
    check("graph Laplacian smallest eigenvalue is 0", abs(small) < 1e-8, f"{small:.2e}")
    # the constant vector is the null eigenvector; the second-smallest (Fiedler) is positive
    two_small = smallest_eigenvalues(laplacian_mv, n, k=2, m=n)
    check("Fiedler value (2nd smallest) is positive", two_small[1] > 1e-6, f"{two_small[1]:.4f}")
    # Laplacian eigenvalues are non-negative
    all_vals = eigenvalues(laplacian_mv, n, m=n)
    check("all Laplacian eigenvalues non-negative", all(v > -1e-8 for v in all_vals))

    # ---- 7. diagonal matrix: Lanczos recovers the DISTINCT eigenvalues ----------------
    # single-start Lanczos finds only distinct eigenvalues (a repeated value appears once), a
    # known property of the Krylov subspace -- so a diagonal with a repeat gives the distinct set.
    diag = [3.0, 1.0, 4.0, 1.0, 5.0]
    dv = matvec_from_matrix([[diag[i] if i == j else 0.0 for j in range(5)] for i in range(5)])
    vals = eigenvalues(dv, 5, m=5)
    distinct = sorted(set(diag))
    check("diagonal matrix yields its distinct eigenvalues",
          all(abs(a - b) < 1e-9 for a, b in zip(sorted(vals), distinct)),
          f"{sorted(vals)} vs {distinct}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
