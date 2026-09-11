"""Tests for regression.py -- linear and logistic regression.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Fits are checked against exact
coefficients, R^2, gradient-descent/closed-form equivalence, and classification accuracy.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import regression as R  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol=1e-4):
    return abs(a - b) <= tol


# --- linear regression: exact line -----------------------------------------
X = [[x] for x in range(10)]
y = [3 * x + 2 for x in range(10)]
w = R.linear_fit(X, y)
check("linear fit recovers slope and intercept", approx(w[0], 2.0) and approx(w[1], 3.0))
check("R^2 is 1 for a perfect fit", approx(R.r_squared(X, y, w), 1.0, 1e-9))
check("MSE is ~0 for a perfect fit", R.mse(X, y, w) < 1e-9)
check("predictions match the targets", all(approx(p, yi) for p, yi in zip(R.predict(X, w), y)))

# --- multivariate linear ----------------------------------------------------
X2 = [[a, b] for a in range(5) for b in range(5)]
y2 = [1 + 2 * a - 3 * b for a, b in X2]
w2 = R.linear_fit(X2, y2)
check("multivariate fit recovers all coefficients",
      approx(w2[0], 1.0) and approx(w2[1], 2.0) and approx(w2[2], -3.0))

# --- gradient descent agrees with the closed form --------------------------
wgd = R.linear_fit_gd(X, y, lr=0.02, epochs=8000)
check("gradient descent matches the QR solution", approx(wgd[0], w[0], 1e-2) and approx(wgd[1], w[1], 1e-2))

# --- noisy data: R^2 high but < 1 ------------------------------------------
def lcg(seed):
    s = seed
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield s >> 8


gen = lcg(1)
noisy_y = [3 * x + 2 + (next(gen) / (1 << 24) - 0.5) * 2 for x in range(30)]
Xn = [[x] for x in range(30)]
wn = R.linear_fit(Xn, noisy_y)
r2 = R.r_squared(Xn, noisy_y, wn)
check("noisy linear fit has high R^2", r2 > 0.98)
check("noisy R^2 is below 1", r2 < 1.0)
check("noisy fit slope is near the true 3", approx(wn[1], 3.0, 0.1))

# --- R^2 sanity -------------------------------------------------------------
# a constant target: any fit that returns the mean has R^2 defined as 1 here (SS_tot=0)
check("constant target R^2 is 1", approx(R.r_squared([[0], [1], [2]], [5, 5, 5], R.linear_fit([[0], [1], [2]], [5, 5, 5])), 1.0))

# --- sigmoid ----------------------------------------------------------------
check("sigmoid(0) is 0.5", approx(R.sigmoid(0), 0.5, 1e-12))
check("sigmoid saturates to 1", approx(R.sigmoid(30), 1.0, 1e-6))
check("sigmoid saturates to 0", approx(R.sigmoid(-30), 0.0, 1e-6))
check("sigmoid is symmetric", approx(R.sigmoid(2) + R.sigmoid(-2), 1.0, 1e-12))

# --- logistic regression: separable 1D -------------------------------------
Xc = [[x] for x in range(11)]
yc = [1 if x > 5 else 0 for x in range(11)]
wl = R.logistic_fit(Xc, yc, lr=0.5, epochs=5000)
check("logistic achieves perfect accuracy on separable data", approx(R.accuracy(Xc, yc, wl), 1.0, 1e-9))
check("logistic log-loss is small", R.log_loss(Xc, yc, wl) < 0.1)
check("decision boundary is near x = 5.5", approx(-wl[0] / wl[1], 5.5, 0.3))
check("predict_proba stays in [0,1]", all(0 <= p <= 1 for p in R.predict_proba(Xc, wl)))
check("predict_class returns 0/1", all(c in (0, 1) for c in R.predict_class(Xc, wl)))
# probabilities are monotone in x for a positive weight
probs = R.predict_proba(Xc, wl)
check("probabilities increase with x", all(probs[i] <= probs[i + 1] + 1e-9 for i in range(len(probs) - 1)))

# --- logistic in 2D ---------------------------------------------------------
X3 = [[0, 0], [1, 1], [0, 1], [1, 0], [5, 5], [6, 5], [5, 6], [6, 6]]
y3 = [0, 0, 0, 0, 1, 1, 1, 1]
wl2 = R.logistic_fit(X3, y3, lr=0.3, epochs=3000)
check("logistic separates two 2D clusters", approx(R.accuracy(X3, y3, wl2), 1.0, 1e-9))

# --- L2 regularization shrinks the weights ---------------------------------
w_plain = R.logistic_fit(Xc, yc, lr=0.5, epochs=3000, l2=0.0)
w_reg = R.logistic_fit(Xc, yc, lr=0.5, epochs=3000, l2=1.0)
check("L2 regularization shrinks the logistic weights", abs(w_reg[1]) < abs(w_plain[1]))
wl_plain = R.linear_fit_gd(X, y, lr=0.02, epochs=3000, l2=0.0)
wl_reg = R.linear_fit_gd(X, y, lr=0.02, epochs=3000, l2=2.0)
check("L2 shrinks the linear weight too", abs(wl_reg[1]) <= abs(wl_plain[1]) + 1e-9)

# --- log-loss decreases during training ------------------------------------
losses = [R.log_loss(Xc, yc, R.logistic_fit(Xc, yc, lr=0.5, epochs=e)) for e in (50, 500, 5000)]
check("log-loss decreases with more epochs", losses[0] > losses[1] > losses[2])

# --- accuracy of a random-guess baseline is ~0.5 on balanced data ----------
zero_w = [0.0, 0.0]
check("all-zero logistic weights give chance accuracy", 0.3 <= R.accuracy(Xc, yc, zero_w) <= 0.7)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall regression tests passed")
