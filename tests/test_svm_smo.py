"""Tests for SVM-SMO: separable margin, KKT, dual constraint, RBF solves XOR/rings, 2-point exact."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from svm_smo import (  # noqa: E402
    SVM,
    linear_kernel,
    rbf_kernel,
    poly_kernel,
    functional_margins,
    dual_constraint,
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
    # ---- 1. linearly separable data: perfect classification ----------------------------
    rng = _lcg(1)
    X, y = [], []
    for _ in range(40):
        # class +1 around (2,2), class -1 around (-2,-2)
        X.append([2 + rng(), 2 + rng()])
        y.append(1)
        X.append([-2 - rng(), -2 - rng()])
        y.append(-1)
    svm = SVM(kernel=linear_kernel, C=10.0, max_passes=100).fit(X, y)
    preds = svm.predict_all(X)
    check("separable: all training labels correct", preds == y)

    # ---- 2. functional margin >= 1 for support vectors (max-margin property) ------------
    fm = functional_margins(svm)
    # all points on the correct side with margin >= ~1 (allow small numerical slack)
    check("separable: all functional margins >= 1 - eps", all(m >= 1 - 1e-2 for m in fm),
          f"min margin {min(fm):.4f}")

    # ---- 3. dual equality constraint sum alpha_i y_i = 0 --------------------------------
    check("dual constraint sum alpha*y = 0", abs(dual_constraint(svm)) < 1e-6,
          f"{dual_constraint(svm):.2e}")

    # ---- 4. at least one support vector, and only SVs have alpha > 0 --------------------
    sv = svm.support_vectors()
    check("has support vectors", len(sv) >= 2, f"{len(sv)}")
    check("non-SVs have alpha ~ 0", all(svm.alpha[i] <= svm.eps for i in range(len(X)) if i not in set(sv)))

    # ---- 5. two-point exact max-margin line ---------------------------------------------
    # points (0,0) label -1 and (2,0) label +1: boundary is the perpendicular bisector x=1,
    # w = (1, 0)*k, margin points both at functional margin 1.
    X2 = [[0.0, 0.0], [2.0, 0.0]]
    y2 = [-1, 1]
    svm2 = SVM(kernel=linear_kernel, C=100.0, max_passes=100).fit(X2, y2)
    # decision boundary should cross near x=1
    # find x where f([x,0]) = 0 by checking sign change
    f0 = svm2.decision_function([0.0, 0.0])
    f2 = svm2.decision_function([2.0, 0.0])
    check("two-point: correct signs", f0 < 0 and f2 > 0, f"{f0:.3f} {f2:.3f}")
    fmid = svm2.decision_function([1.0, 0.0])
    check("two-point: boundary at x=1", abs(fmid) < 1e-6, f"f(1,0)={fmid:.3e}")
    # weight vector points along +x
    w = svm2.weight_vector()
    check("two-point: weight along +x", w[0] > 0 and abs(w[1]) < 1e-6, f"{w}")

    # ---- 6. weight vector matches analytic max-margin normal ----------------------------
    # KKT at the margin: w.x+ + b = 1, w.x- + b = -1 => w.(x+ - x-) = 2. Here x+ - x- = (2,0),
    # so w = (1, 0): w0 * 2 = 2 => w0 = 1. (Margin half-width 1/|w| = 1, correct for points 2 apart.)
    check("two-point: w0 = 1", abs(w[0] - 1.0) < 1e-3, f"w0={w[0]:.4f}")

    # ---- 7. RBF kernel solves XOR (not linearly separable) ------------------------------
    Xxor = [[0, 0], [1, 1], [0, 1], [1, 0]]
    yxor = [1, 1, -1, -1]
    svm_rbf = SVM(kernel=rbf_kernel(gamma=2.0), C=100.0, max_passes=200).fit(Xxor, yxor)
    check("RBF solves XOR", svm_rbf.predict_all(Xxor) == yxor, f"{svm_rbf.predict_all(Xxor)}")

    # ---- 8. RBF separates two concentric rings ------------------------------------------
    rng2 = _lcg(7)
    Xr, yr = [], []
    for _ in range(30):
        # inner ring radius ~1 -> class +1
        th = 2 * math.pi * rng2()
        r = 0.9 + 0.15 * rng2()
        Xr.append([r * math.cos(th), r * math.sin(th)])
        yr.append(1)
        # outer ring radius ~3 -> class -1
        th = 2 * math.pi * rng2()
        r = 2.8 + 0.15 * rng2()
        Xr.append([r * math.cos(th), r * math.sin(th)])
        yr.append(-1)
    svm_ring = SVM(kernel=rbf_kernel(gamma=0.5), C=100.0, max_passes=200).fit(Xr, yr)
    acc = sum(1 for i in range(len(Xr)) if svm_ring.predict(Xr[i]) == yr[i]) / len(Xr)
    check("RBF separates concentric rings", acc >= 0.95, f"acc {acc:.3f}")

    # ---- 9. polynomial kernel also solves XOR -------------------------------------------
    svm_poly = SVM(kernel=poly_kernel(degree=2, gamma=1.0, coef0=1.0), C=100.0,
                   max_passes=200).fit(Xxor, yxor)
    check("poly kernel solves XOR", svm_poly.predict_all(Xxor) == yxor)

    # ---- 10. alphas stay in [0, C] ------------------------------------------------------
    check("alphas within [0, C]", all(-1e-9 <= a <= svm.C + 1e-9 for a in svm.alpha))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
