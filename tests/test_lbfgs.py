"""Tests for lbfgs: known optima, gradient checks, vs gradient descent, logistic regression."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lbfgs import minimize, finite_diff_gradient

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- quadratic bowl: exact minimum at the origin ---------------------------
sphere = lambda x: sum(v * v for v in x)
sphere_g = lambda x: [2 * v for v in x]
r = minimize(sphere, [3.0, -2.0, 1.5, 0.8], grad=sphere_g)
check("sphere converges", r["converged"] and r["fx"] < 1e-12)
check("sphere reaches the origin", all(abs(v) < 1e-6 for v in r["x"]))

# --- finite-difference gradient matches the analytic one -------------------
x = [0.3, -1.1, 2.0]
fd = finite_diff_gradient(sphere, x)
an = sphere_g(x)
check("finite-diff gradient matches analytic (sphere)",
      all(abs(fd[i] - an[i]) < 1e-5 for i in range(3)))


def rosen(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


def rosen_g(x):
    n = len(x)
    g = [0.0] * n
    for i in range(n - 1):
        g[i] += -400 * x[i] * (x[i + 1] - x[i] ** 2) - 2 * (1 - x[i])
        g[i + 1] += 200 * (x[i + 1] - x[i] ** 2)
    return g


# gradient check on Rosenbrock at a non-trivial point
xr = [0.5, -0.7, 1.3]
fd = finite_diff_gradient(rosen, xr)
an = rosen_g(xr)
check("finite-diff gradient matches analytic (Rosenbrock)",
      all(abs(fd[i] - an[i]) < 1e-3 for i in range(3)))

# --- Rosenbrock: optimum at all-ones ---------------------------------------
r = minimize(rosen, [-1.2, 1.0], grad=rosen_g, max_iter=2000)
check("Rosenbrock 2D converges", r["fx"] < 1e-8)
check("Rosenbrock 2D solution near (1,1)", all(abs(v - 1) < 1e-3 for v in r["x"]))

r = minimize(rosen, [-1.0, -1.0, -1.0, -1.0], grad=rosen_g, max_iter=5000)
check("Rosenbrock 4D converges", r["fx"] < 1e-6)

# --- works with the automatic finite-difference gradient (no analytic grad) -
r = minimize(rosen, [-1.2, 1.0], max_iter=2000)
check("Rosenbrock via finite-difference gradient converges", r["fx"] < 1e-6)

# --- shifted optimum -------------------------------------------------------
shift = [2.5, -1.5, 3.0, 0.5]
shifted = lambda x: sum((x[i] - shift[i]) ** 2 for i in range(len(x)))
shifted_g = lambda x: [2 * (x[i] - shift[i]) for i in range(len(x))]
r = minimize(shifted, [0.0] * 4, grad=shifted_g)
check("shifted quadratic recovers the optimum",
      all(abs(r["x"][i] - shift[i]) < 1e-5 for i in range(4)))

# --- ill-conditioned quadratic: L-BFGS beats gradient descent --------------
# f(x) = sum a_i x_i^2 with a spread of scales
scales = [1.0, 10.0, 100.0, 1000.0]
def illcond(x):
    return sum(scales[i] * x[i] ** 2 for i in range(len(x)))
def illcond_g(x):
    return [2 * scales[i] * x[i] for i in range(len(x))]


def gradient_descent(f, g, x0, lr, iters):
    x = list(x0)
    for _ in range(iters):
        grad = g(x)
        x = [x[i] - lr * grad[i] for i in range(len(x))]
    return f(x)


x0 = [1.0, 1.0, 1.0, 1.0]
lbfgs_r = minimize(illcond, x0, grad=illcond_g, max_iter=200)
# gradient descent must use a tiny lr for stability on the 1000-scale direction
gd_val = gradient_descent(illcond, illcond_g, x0, lr=1.0 / (2 * 1000), iters=200)
check(f"L-BFGS beats gradient descent on ill-conditioned quadratic "
      f"({lbfgs_r['fx']:.2e} vs {gd_val:.2e})", lbfgs_r["fx"] < gd_val)
check("L-BFGS solves ill-conditioned quadratic to high precision", lbfgs_r["fx"] < 1e-10)

# --- logistic regression: recover a separating weight vector ---------------
# generate a linearly separable dataset and fit by minimizing the NLL
state = 5
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0

true_w = [1.5, -2.0]
true_b = 0.5
X = []
Y = []
for _ in range(200):
    xi = [rng() * 4 - 2, rng() * 4 - 2]
    z = true_w[0] * xi[0] + true_w[1] * xi[1] + true_b
    p = 1 / (1 + math.exp(-z))
    y = 1 if rng() < p else 0
    X.append(xi)
    Y.append(y)


def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def nll(w):
    # w = [w0, w1, b]; add small L2 for well-posedness
    total = 0.0
    for xi, y in zip(X, Y):
        z = w[0] * xi[0] + w[1] * xi[1] + w[2]
        p = sigmoid(z)
        p = min(max(p, 1e-12), 1 - 1e-12)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    total += 1e-3 * sum(wi * wi for wi in w)
    return total


def nll_g(w):
    g = [0.0, 0.0, 0.0]
    for xi, y in zip(X, Y):
        z = w[0] * xi[0] + w[1] * xi[1] + w[2]
        err = sigmoid(z) - y
        g[0] += err * xi[0]
        g[1] += err * xi[1]
        g[2] += err
    g[0] += 2e-3 * w[0]
    g[1] += 2e-3 * w[1]
    g[2] += 2e-3 * w[2]
    return g


r = minimize(nll, [0.0, 0.0, 0.0], grad=nll_g, max_iter=500)
check("logistic regression converges", r["converged"] or r["grad_norm"] < 1e-4)
# check classification accuracy on the training set
w = r["x"]
correct = 0
for xi, y in zip(X, Y):
    pred = 1 if (w[0] * xi[0] + w[1] * xi[1] + w[2]) > 0 else 0
    if pred == y:
        correct += 1
acc = correct / len(Y)
check(f"logistic-regression weights classify the data well (acc {acc:.2f})", acc > 0.80)
# recovered weight direction aligns with the true direction (cosine similarity)
def cos(a, b):
    return sum(a[i] * b[i] for i in range(len(a))) / (
        math.sqrt(sum(v * v for v in a)) * math.sqrt(sum(v * v for v in b)))
check("recovered weight direction aligns with the true weights",
      cos(w[:2], true_w) > 0.9)

# --- gradient-check on the logistic NLL ------------------------------------
wtest = [0.4, -0.6, 0.2]
fd = finite_diff_gradient(nll, wtest, h=1e-6)
an = nll_g(wtest)
check("finite-diff gradient matches analytic (logistic NLL)",
      all(abs(fd[i] - an[i]) < 1e-2 for i in range(3)))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all lbfgs tests passed")
