"""Tests for Sinkhorn OT: marginals match, converges to exact 1-D transport, symmetry, self=0."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sinkhorn import (  # noqa: E402
    sinkhorn,
    cost_matrix,
    transport_cost,
    sinkhorn_distance,
    wasserstein_1d,
    exact_transport_1d,
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


def main():
    # ---- 1. plan marginals match the target distributions -------------------------------
    rng = _lcg(2024)
    ok_row = ok_col = True
    for _ in range(50):
        n, m = 4 + int(rng() * 4), 4 + int(rng() * 4)
        a = [rng() + 0.1 for _ in range(n)]
        b = [rng() + 0.1 for _ in range(m)]
        xs = [rng() * 10 for _ in range(n)]
        ys = [rng() * 10 for _ in range(m)]
        C = cost_matrix(xs, ys)
        P, _, _ = sinkhorn(a, b, C, eps=0.1, max_iter=3000)
        sa = sum(a)
        sb = sum(b)
        row = [sum(P[i]) for i in range(n)]
        col = [sum(P[i][j] for i in range(n)) for j in range(m)]
        if any(abs(row[i] - a[i] / sa) > 1e-3 for i in range(n)):
            ok_row = False
        if any(abs(col[j] - b[j] / sb) > 1e-3 for j in range(m)):
            ok_col = False
    check("plan row sums match a", ok_row)
    check("plan column sums match b", ok_col)

    # ---- 2. regularized cost approaches exact 1-D transport as eps shrinks ---------------
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    a = [0.4, 0.3, 0.2, 0.1, 0.0]
    b = [0.0, 0.1, 0.2, 0.3, 0.4]
    C = cost_matrix(xs, xs, p=2)
    exact = exact_transport_1d(a, xs, b, xs, p=2)
    c_big = sinkhorn(a, b, C, eps=1.0, max_iter=5000)[1]
    c_small = sinkhorn(a, b, C, eps=0.02, max_iter=20000)[1]
    check("smaller eps closer to exact transport", abs(c_small - exact) < abs(c_big - exact),
          f"eps1 {c_big:.3f}, eps.02 {c_small:.3f}, exact {exact:.3f}")
    check("small-eps Sinkhorn near exact", abs(c_small - exact) < 0.05, f"{c_small:.4f} vs {exact:.4f}")

    # ---- 3. transporting a distribution to itself costs ~0 ------------------------------
    a = [0.2, 0.3, 0.5]
    xs = [0.0, 1.0, 2.0]
    C = cost_matrix(xs, xs)
    cost = sinkhorn(a, a, C, eps=0.05, max_iter=10000)[1]
    check("self-transport cost ~ 0", cost < 0.02, f"{cost:.4f}")

    # ---- 4. symmetry: cost(a,b) == cost(b,a) --------------------------------------------
    a = [0.5, 0.3, 0.2]
    b = [0.1, 0.4, 0.5]
    xs = [0.0, 1.0, 2.0]
    C = cost_matrix(xs, xs)
    cab = sinkhorn(a, b, C, eps=0.05, max_iter=10000)[1]
    cba = sinkhorn(b, a, C, eps=0.05, max_iter=10000)[1]
    check("transport cost symmetric", abs(cab - cba) < 1e-3, f"{cab:.5f} vs {cba:.5f}")

    # ---- 5. moving a spike costs the ground distance ------------------------------------
    # all mass at x=0 -> all mass at x=5, squared cost should be 25
    a = [1.0, 0.0, 0.0]
    b = [0.0, 0.0, 1.0]
    xs = [0.0, 2.5, 5.0]
    C = cost_matrix(xs, xs, p=2)
    cost = sinkhorn(a, b, C, eps=0.05, max_iter=10000)[1]
    check("spike transport cost ~ ground distance^2 (25)", abs(cost - 25) < 0.5, f"{cost:.3f}")

    # ---- 6. exact 1-D Wasserstein closed form vs greedy transport -----------------------
    # p=1 Wasserstein from CDF integral should match greedy monotone transport (p=1)
    rng = _lcg(77)
    ok = True
    for _ in range(50):
        n = 3 + int(rng() * 4)
        xs = sorted(set(rng() * 10 for _ in range(n)))
        n = len(xs)
        a = [rng() + 0.1 for _ in range(n)]
        b = [rng() + 0.1 for _ in range(n)]
        w = wasserstein_1d(a, xs, b, xs, p=1)
        g = exact_transport_1d(a, xs, b, xs, p=1)
        if abs(w - g) > 1e-6:
            ok = False
    check("1-D Wasserstein (CDF) == greedy monotone transport (p=1)", ok)

    # ---- 7. sinkhorn_distance convenience ------------------------------------------------
    d = sinkhorn_distance([0.5, 0.5], [0.5, 0.5], [0.0, 1.0], eps=0.05)
    check("sinkhorn_distance self ~ 0", d < 0.02, f"{d:.4f}")

    # ---- 8. edge cases ------------------------------------------------------------------
    try:
        sinkhorn([0, 0], [1, 1], [[1, 1], [1, 1]])
        check("zero-mass distribution raises", False)
    except ValueError:
        check("zero-mass distribution raises", True)
    try:
        sinkhorn([1, 1], [1, 1], [[1, 1]])  # wrong shape
        check("shape mismatch raises", False)
    except ValueError:
        check("shape mismatch raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
