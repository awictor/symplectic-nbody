"""Tests for Lasso: recovers sparse support, zeros irrelevant features, lambda sparsity, vs ridge."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lasso import (  # noqa: E402
    lasso,
    predict,
    lasso_path,
    n_nonzero,
    mse,
    ridge,
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


def _sparse_data(n, p, true_coef, rng, noise=0.0):
    X = [[rng() * 2 - 1 for _ in range(p)] for _ in range(n)]
    y = [sum(X[i][j] * true_coef[j] for j in range(p)) + 5 + (rng() - 0.5) * noise
         for i in range(n)]
    return X, y


def main():
    # ---- 1. recovers a sparse support (zeros irrelevant features) -----------------------
    rng = _lcg(2024)
    p = 10
    true_coef = [0.0] * p
    true_coef[2] = 3.0
    true_coef[5] = -2.0
    true_coef[8] = 1.5
    X, y = _sparse_data(80, p, true_coef, rng)
    intercept, coef = lasso(X, y, lam=0.05)
    active = set(j for j in range(p) if abs(coef[j]) > 0.1)
    true_active = {2, 5, 8}
    check("Lasso recovers the true active set", active == true_active,
          f"active {sorted(active)} vs {sorted(true_active)}")
    # nonzero coefficients close to the truth
    check("active coefficients near true values",
          all(abs(coef[j] - true_coef[j]) < 0.4 for j in true_active),
          f"{[round(coef[j],2) for j in sorted(true_active)]}")

    # ---- 2. irrelevant features exactly zero --------------------------------------------
    zero_feats = [j for j in range(p) if j not in {2, 5, 8}]
    check("irrelevant features driven to zero", all(abs(coef[j]) < 0.05 for j in zero_feats),
          f"max irrelevant {max(abs(coef[j]) for j in zero_feats):.4f}")

    # ---- 3. larger lambda -> sparser model ----------------------------------------------
    rng = _lcg(77)
    X, y = _sparse_data(60, 12, [0.0, 4.0, 0.0, -3.0, 0.0, 2.0] + [0.0] * 6, rng)
    path = lasso_path(X, y, [0.01, 0.05, 0.2, 0.5, 1.0])
    nzs = [row[3] for row in path]
    check("sparsity increases (nonzeros decrease) with lambda",
          all(nzs[i] >= nzs[i + 1] for i in range(len(nzs) - 1)), f"{nzs}")

    # ---- 4. lambda = 0 reduces to ordinary least squares --------------------------------
    # small clean problem; lasso(lam=0) should fit like OLS (low error)
    rng = _lcg(7)
    true = [2.0, -1.0, 3.0]
    X, y = _sparse_data(50, 3, true, rng)
    intercept, coef = lasso(X, y, lam=0.0)
    check("lambda=0: recovers dense OLS coefficients",
          all(abs(coef[j] - true[j]) < 0.1 for j in range(3)), f"{[round(c,2) for c in coef]}")
    check("lambda=0: near-zero training error", mse(intercept, coef, X, y) < 0.01)

    # ---- 5. predictions accurate on the active features ---------------------------------
    rng = _lcg(11)
    true_coef = [0.0, 5.0, 0.0, 0.0, -3.0]
    X, y = _sparse_data(100, 5, true_coef, rng, noise=0.1)
    intercept, coef = lasso(X, y, lam=0.02)
    check("Lasso predictions accurate", mse(intercept, coef, X, y) < 0.5, f"{mse(intercept, coef, X, y):.4f}")

    # ---- 6. Lasso is sparse where ridge is dense ----------------------------------------
    rng = _lcg(555)
    true_coef = [0.0, 0.0, 4.0, 0.0, 0.0, -2.5, 0.0, 0.0]
    X, y = _sparse_data(70, 8, true_coef, rng)
    _, lasso_coef = lasso(X, y, lam=0.1)
    _, ridge_coef = ridge(X, y, lam=1.0)
    lasso_nz = n_nonzero(lasso_coef)
    ridge_nz = n_nonzero(ridge_coef)
    check("Lasso sparser than ridge", lasso_nz < ridge_nz, f"lasso {lasso_nz} vs ridge {ridge_nz}")
    check("Lasso finds ~2 active features", lasso_nz <= 3, f"{lasso_nz}")
    check("ridge keeps all features", ridge_nz == 8, f"{ridge_nz}")

    # ---- 7. huge lambda -> all zeros (intercept only) -----------------------------------
    intercept, coef = lasso(X, y, lam=100.0)
    check("huge lambda zeros all coefficients", n_nonzero(coef) == 0)
    # intercept should be the mean of y
    check("huge lambda intercept = mean(y)", abs(intercept - sum(y) / len(y)) < 1e-6)

    # ---- 8. edge case: single feature ---------------------------------------------------
    rng = _lcg(3)
    X = [[rng()] for _ in range(30)]
    y = [3 * X[i][0] + 1 for i in range(30)]
    intercept, coef = lasso(X, y, lam=0.001)
    check("single feature slope ~ 3", abs(coef[0] - 3) < 0.2, f"{coef[0]:.3f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
