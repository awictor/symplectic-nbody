"""Tests for conjugate_gradient.py -- iterative SPD solver.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. CG is cross-checked against a
dense LU solve, the residual decay, and the <= n step guarantee.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import conjugate_gradient as CG  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def vclose(u, v, tol=1e-6):
    return all(abs(a - b) <= tol for a, b in zip(u, v))


# --- a small SPD system -----------------------------------------------------
A = [[4, 1], [1, 3]]
b = [1, 2]
x, hist = CG.conjugate_gradient(A, b)
ref = CG.spd_reference(A, b)
check("CG matches the LU solution", vclose(x, ref, 1e-9))
check("CG residual is ~0", CG.residual_norm(A, x, b) < 1e-9)
check("residual history is recorded", len(hist) >= 1)
check("residual decreases monotonically", all(hist[i] >= hist[i + 1] - 1e-12 for i in range(len(hist) - 1)))
check("solve wrapper returns the solution", vclose(CG.solve(A, b), ref, 1e-9))

# --- exact solution on a diagonal system -----------------------------------
D = [[2, 0, 0], [0, 5, 0], [0, 0, 10]]
check("CG solves a diagonal SPD system", vclose(CG.conjugate_gradient(D, [4, 5, 20])[0], [2, 1, 2], 1e-9))

# --- converges in at most n steps ------------------------------------------
for n in (5, 10, 20):
    Asp = CG.make_spd(n, seed=n)
    bsp = [float(i + 1) for i in range(n)]
    xs, h = CG.conjugate_gradient(Asp, bsp)
    check(f"CG on n={n} converges in <= n steps", len(h) - 1 <= n)
    check(f"CG on n={n} matches LU", vclose(xs, CG.spd_reference(Asp, bsp), 1e-5))
    check(f"CG on n={n} drives the residual to ~0", CG.residual_norm(Asp, xs, bsp) < 1e-5)

# --- CG minimizes the quadratic energy -------------------------------------
Asp = CG.make_spd(6, seed=3)
bsp = [float(i) for i in range(6)]
xs, _ = CG.conjugate_gradient(Asp, bsp)
# the energy at the CG solution is lower than at a perturbed point
worse = [xs[i] + 0.3 for i in range(6)]
check("CG solution minimizes the energy", CG.energy(Asp, xs, bsp) < CG.energy(Asp, worse, bsp))

# --- preconditioned CG (Jacobi) --------------------------------------------
for n in (4, 15, 30):
    Asp = CG.make_spd(n, seed=n + 1)
    bsp = [float(i + 1) for i in range(n)]
    xp, hp = CG.preconditioned_cg(Asp, bsp)
    check(f"PCG on n={n} matches LU", vclose(xp, CG.spd_reference(Asp, bsp), 1e-5))
    check(f"PCG on n={n} drives the residual to ~0", CG.residual_norm(Asp, xp, bsp) < 1e-5)
try:
    CG.preconditioned_cg([[0, 1], [1, 0]], [1, 1])   # zero diagonal
    check("PCG rejects a zero diagonal", False)
except ValueError:
    check("PCG rejects a zero diagonal", True)

# --- CG only uses mat-vecs: works on a large sparse-ish SPD matrix ---------
big = CG.make_spd(40, seed=7)
bbig = [math.sin(i) for i in range(40)]
xb, hb = CG.conjugate_gradient(big, bbig)
check("CG solves a 40x40 SPD system", CG.residual_norm(big, xb, bbig) < 1e-5)
check("CG on 40x40 converges within n steps", len(hb) - 1 <= 40)

# --- helpers ----------------------------------------------------------------
check("is_symmetric accepts a symmetric matrix", CG.is_symmetric([[1, 2], [2, 1]]))
check("is_symmetric rejects a non-symmetric matrix", not CG.is_symmetric([[1, 2], [3, 4]]))
check("make_spd is symmetric", CG.is_symmetric(CG.make_spd(5, seed=1)))
# make_spd is positive definite: the energy has a unique minimum, so CG converges
sp = CG.make_spd(5, seed=9)
check("make_spd is solvable by CG (hence SPD)", CG.residual_norm(sp, CG.solve(sp, [1, 2, 3, 4, 5]), [1, 2, 3, 4, 5]) < 1e-5)

# --- residual history strictly informative --------------------------------
_, h = CG.conjugate_gradient(CG.make_spd(8, seed=2), [1.0] * 8)
check("first residual is ||b|| when starting from zero", abs(h[0] - math.sqrt(8)) < 1e-9)
check("final residual is far below the first", h[-1] < 1e-6 * h[0])


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall conjugate_gradient tests passed")
