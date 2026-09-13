"""Tests for OMP: recovers planted sparse signal, monotone residual, exact square solve, noise."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from omp import (  # noqa: E402
    omp,
    random_matrix,
    matvec,
    residual_norm,
    support,
)


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


def _sparse_signal(n, k, rng):
    """A k-sparse signal: k random nonzero coefficients."""
    x = [0.0] * n
    idxs = set()
    while len(idxs) < k:
        idxs.add(int(rng() * n))
    for j in idxs:
        x[j] = rng() * 4 - 2
        if abs(x[j]) < 0.3:
            x[j] += 1.0  # keep coefficients well above zero
    return x


def main():
    # ---- 1. recover a planted sparse signal from few measurements -----------------------
    rng = _lcg(2024)
    successes = 0
    trials = 40
    for t in range(trials):
        n, k = 40, 3
        m = 20  # m ~ several times k
        x_true = _sparse_signal(n, k, rng)
        A = random_matrix(m, n, seed=100 + t)
        y = matvec(A, x_true)
        x_rec = omp(A, y, sparsity=k)
        # exact support recovery + coefficient match
        if support(x_rec) == support(x_true) and \
           all(abs(x_rec[i] - x_true[i]) < 1e-6 for i in range(n)):
            successes += 1
    check("OMP recovers planted k-sparse signal", successes >= trials - 2,
          f"{successes}/{trials}")

    # ---- 2. residual decreases monotonically --------------------------------------------
    # instrument by running increasing sparsity budgets
    rng = _lcg(77)
    n, k, m = 50, 5, 30
    x_true = _sparse_signal(n, k, rng)
    A = random_matrix(m, n, seed=7)
    y = matvec(A, x_true)
    prev = float("inf")
    ok = True
    for budget in range(1, k + 1):
        x = omp(A, y, sparsity=budget)
        r = residual_norm(A, x, y)
        if r > prev + 1e-9:
            ok = False
        prev = r
    check("residual decreases with sparsity budget", ok)
    check("residual ~0 at true sparsity", residual_norm(A, omp(A, y, sparsity=k), y) < 1e-6)

    # ---- 3. stops at the right sparsity -------------------------------------------------
    x = omp(A, y, sparsity=k)
    check("OMP uses at most k atoms", len(support(x)) <= k)

    # ---- 4. fully-measured square invertible system solved exactly ----------------------
    rng = _lcg(11)
    n = 8
    A = random_matrix(n, n, seed=3, normalize_cols=False)
    x_true = [rng() * 2 - 1 for _ in range(n)]
    y = matvec(A, x_true)
    x_rec = omp(A, y, sparsity=n, tol=1e-12)
    check("square system solved exactly", all(abs(x_rec[i] - x_true[i]) < 1e-6 for i in range(n)),
          f"max err {max(abs(x_rec[i]-x_true[i]) for i in range(n)):.2e}")

    # ---- 5. graceful degradation under measurement noise --------------------------------
    rng = _lcg(555)
    n, k, m = 40, 3, 24
    x_true = _sparse_signal(n, k, rng)
    A = random_matrix(m, n, seed=9)
    y_clean = matvec(A, x_true)
    noise = [(rng() - 0.5) * 0.02 for _ in range(m)]
    y = [y_clean[i] + noise[i] for i in range(m)]
    x_rec = omp(A, y, sparsity=k)
    # support usually still recovered; coefficients close
    err = max(abs(x_rec[i] - x_true[i]) for i in range(n))
    check("bounded error under small noise", err < 0.1, f"max err {err:.4f}")

    # ---- 6. tolerance-based stopping ----------------------------------------------------
    rng = _lcg(13)
    n, k, m = 30, 4, 22
    x_true = _sparse_signal(n, k, rng)
    A = random_matrix(m, n, seed=17)
    y = matvec(A, x_true)
    x_rec = omp(A, y, tol=1e-9, max_iter=m)  # no sparsity given, stop on residual
    check("tol-based OMP recovers support", support(x_rec) == support(x_true))

    # ---- 7. edge cases ------------------------------------------------------------------
    # zero signal -> zero recovery
    A = random_matrix(10, 20, seed=1)
    x = omp(A, [0.0] * 10, sparsity=3)
    check("zero measurements -> zero (or near-zero) recovery", residual_norm(A, x, [0.0] * 10) < 1e-9)
    # single atom
    A = random_matrix(10, 20, seed=2)
    x_true = [0.0] * 20
    x_true[5] = 3.0
    y = matvec(A, x_true)
    x = omp(A, y, sparsity=1)
    check("single-atom recovery", support(x) == [5] and abs(x[5] - 3.0) < 1e-6)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
