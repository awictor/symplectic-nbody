"""Tests for NMF: recovers planted factors, monotone error decrease, non-negativity, KL objective."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nmf import (  # noqa: E402
    nmf,
    nmf_kl,
    frobenius_error,
    kl_divergence,
    reconstruct,
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


def _matmul(A, B):
    return [[sum(A[i][t] * B[t][j] for t in range(len(B)))
             for j in range(len(B[0]))] for i in range(len(A))]


def main():
    # ---- 1. recover a planted low-rank non-negative factorization -----------------------
    rng = _lcg(2024)
    m, n, k = 8, 6, 2
    Wt = [[rng() for _ in range(k)] for _ in range(m)]
    Ht = [[rng() for _ in range(n)] for _ in range(k)]
    V = _matmul(Wt, Ht)
    W, H, curve = nmf(V, k, iterations=400, seed=7, track_error=True)
    err = frobenius_error(V, W, H)
    Vnorm = frobenius_error(V, [[0] * k for _ in range(m)], [[0] * n for _ in range(k)])
    check("recovers planted rank-2 factorization (small error)", err < 0.02 * Vnorm,
          f"err {err:.4f} vs |V| {Vnorm:.2f}")

    # ---- 2. reconstruction reproduces V -------------------------------------------------
    WH = reconstruct(W, H)
    maxdiff = max(abs(V[i][j] - WH[i][j]) for i in range(m) for j in range(n))
    check("W H reproduces V entrywise", maxdiff < 0.05, f"max diff {maxdiff:.4f}")

    # ---- 3. Frobenius error decreases monotonically (Lee-Seung guarantee) ---------------
    mono = all(curve[i + 1] <= curve[i] + 1e-9 for i in range(len(curve) - 1))
    check("Frobenius error monotonically non-increasing", mono)
    check("error curve actually decreased", curve[-1] < curve[0])

    # ---- 4. factors stay non-negative ---------------------------------------------------
    nn = all(W[i][a] >= 0 for i in range(m) for a in range(k)) and \
         all(H[a][j] >= 0 for a in range(k) for j in range(n))
    check("W and H are non-negative", nn)

    # ---- 5. higher rank fits at least as well -------------------------------------------
    rng = _lcg(99)
    V2 = [[rng() for _ in range(6)] for _ in range(7)]
    _, _, c1 = nmf(V2, 1, iterations=300, seed=3, track_error=True)
    _, _, c3 = nmf(V2, 3, iterations=300, seed=3, track_error=True)
    check("rank-3 error <= rank-1 error", c3[-1] <= c1[-1] + 1e-6, f"{c3[-1]:.3f} vs {c1[-1]:.3f}")

    # ---- 6. exact rank-1 non-negative matrix factors perfectly --------------------------
    # V = outer(u, v), u,v >= 0
    u = [1.0, 2.0, 3.0]
    v = [4.0, 1.0, 0.5, 2.0]
    V1 = [[u[i] * v[j] for j in range(4)] for i in range(3)]
    W, H = nmf(V1, 1, iterations=500, seed=1)
    check("rank-1 outer product factors near-exactly", frobenius_error(V1, W, H) < 1e-3,
          f"{frobenius_error(V1, W, H):.5f}")

    # ---- 7. KL divergence objective decreases -------------------------------------------
    rng = _lcg(555)
    Vc = [[float(1 + int(rng() * 10)) for _ in range(5)] for _ in range(6)]  # count-like
    _, _, klcurve = nmf_kl(Vc, 2, iterations=200, seed=4, track_error=True)
    kl_mono = all(klcurve[i + 1] <= klcurve[i] + 1e-6 for i in range(len(klcurve) - 1))
    check("KL divergence monotonically non-increasing", kl_mono)
    check("KL curve decreased", klcurve[-1] < klcurve[0])

    # ---- 8. errors/edge cases -----------------------------------------------------------
    try:
        nmf([[1, -1], [2, 3]], 1)
        check("negative input raises", False)
    except ValueError:
        check("negative input raises", True)
    # a zero matrix factors to (near) zero error
    Z = [[0.0] * 3 for _ in range(3)]
    W, H = nmf(Z, 2, iterations=50, seed=2)
    check("zero matrix reconstructs to near zero", frobenius_error(Z, W, H) < 1e-3)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
