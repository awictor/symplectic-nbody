"""Tests for cma_es: convergence on benchmarks, vs random search, rotation invariance, shifted optima."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cma_es import minimize, sphere, rosenbrock, rastrigin, ellipsoid, _eig_symmetric

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- Jacobi eigensolver sanity ---------------------------------------------
# a known symmetric matrix: [[2,1],[1,2]] has eigenvalues 1 and 3
evals, V = _eig_symmetric([[2.0, 1.0], [1.0, 2.0]])
check("Jacobi eigenvalues of [[2,1],[1,2]] are {1,3}",
      abs(min(evals) - 1) < 1e-9 and abs(max(evals) - 3) < 1e-9)
# eigenvectors orthonormal: V^T V = I
n = 2
ok = True
for i in range(n):
    for j in range(n):
        dot = sum(V[k][i] * V[k][j] for k in range(n))
        if abs(dot - (1 if i == j else 0)) > 1e-9:
            ok = False
check("Jacobi eigenvectors are orthonormal", ok)
# reconstruction: V diag(evals) V^T == A
A = [[2.0, 1.0], [1.0, 2.0]]
recon_ok = True
for i in range(n):
    for j in range(n):
        s = sum(V[i][k] * evals[k] * V[j][k] for k in range(n))
        if abs(s - A[i][j]) > 1e-9:
            recon_ok = False
check("Jacobi decomposition reconstructs the matrix", recon_ok)

# --- sphere: converges to 0 at the origin ----------------------------------
r = minimize(sphere, [3.0, -2.0, 1.5, 0.8], sigma0=0.5, seed=1)
check("sphere converges below 1e-9", r["fx"] < 1e-9)
check("sphere solution is near the origin", all(abs(xi) < 1e-4 for xi in r["x"]))

# --- Rosenbrock: the classic curved valley, optimum at (1,1,...) -----------
r = minimize(rosenbrock, [0.0, 0.0], sigma0=0.3, max_iter=3000, seed=1)
check("Rosenbrock 2D converges below 1e-6", r["fx"] < 1e-6)
check("Rosenbrock 2D solution near (1,1)", all(abs(xi - 1) < 1e-2 for xi in r["x"]))

r3 = minimize(rosenbrock, [0.0, 0.0, 0.0], sigma0=0.3, max_iter=4000, seed=2)
check("Rosenbrock 3D converges below 1e-4", r3["fx"] < 1e-4)

# --- Rastrigin: highly multimodal, CMA-ES with a decent start finds global -
best = float("inf")
for seed in range(1, 6):
    r = minimize(rastrigin, [0.2, -0.3], sigma0=0.3, max_iter=2000, seed=seed)
    best = min(best, r["fx"])
check("Rastrigin 2D reaches near 0 for some seed", best < 1e-4)

# --- ellipsoid: badly conditioned (1e6 spread) -- covariance adaptation wins -
r = minimize(ellipsoid, [1.0, 1.0, 1.0], sigma0=0.3, max_iter=3000, seed=1)
check("ill-conditioned ellipsoid converges below 1e-6", r["fx"] < 1e-6)

# --- shifted optimum: minimum away from the origin -------------------------
shift = [2.5, -1.5, 3.0]
def shifted(x):
    return sum((x[i] - shift[i]) ** 2 for i in range(len(x)))

r = minimize(shifted, [0.0, 0.0, 0.0], sigma0=0.5, seed=1)
check("shifted-sphere converges below 1e-9", r["fx"] < 1e-9)
check("shifted-sphere recovers the shifted optimum",
      all(abs(r["x"][i] - shift[i]) < 1e-4 for i in range(3)))

# --- vastly outperforms random search under an equal budget ----------------
def random_search(f, x0, span, evals, seed):
    state = seed
    def u():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    best = float("inf")
    n = len(x0)
    for _ in range(evals):
        x = [x0[i] + (u() * 2 - 1) * span for i in range(n)]
        best = min(best, f(x))
    return best

cma = minimize(sphere, [3.0, 3.0, 3.0, 3.0], sigma0=0.5, seed=1)
rs = random_search(sphere, [3.0, 3.0, 3.0, 3.0], 6.0, cma["evaluations"], seed=1)
check(f"CMA-ES beats random search on sphere ({cma['fx']:.2e} vs {rs:.2e})", cma["fx"] < rs * 1e-3)

# --- rotation invariance: rotating the sphere doesn't change performance ----
# rotate coordinates by an orthogonal matrix; the sphere is symmetric, so CMA-ES should still
# converge (a hallmark: CMA-ES is invariant to rotations of the search space).
theta = 0.7
def rotated_ellipse(x):
    # a 2D anisotropic bowl rotated by theta -- ill-scaled but smooth
    c, s = math.cos(theta), math.sin(theta)
    u = c * x[0] - s * x[1]
    v = s * x[0] + c * x[1]
    return u * u + 100 * v * v

r = minimize(rotated_ellipse, [1.0, 1.0], sigma0=0.3, max_iter=2000, seed=1)
check("rotated anisotropic bowl converges below 1e-8", r["fx"] < 1e-8)

# --- reproducibility: same seed -> same result -----------------------------
a = minimize(sphere, [1.0, 2.0], sigma0=0.4, seed=7)
b = minimize(sphere, [1.0, 2.0], sigma0=0.4, seed=7)
check("same seed gives identical result", abs(a["fx"] - b["fx"]) < 1e-18 and a["x"] == b["x"])

# --- returns a sane evaluation count ---------------------------------------
check("evaluation count is positive and finite", 0 < r["evaluations"] < 10 ** 7)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all cma_es tests passed")
