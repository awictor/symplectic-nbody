"""Tests for newton_nd: nonlinear systems, quadratic convergence, FD Jacobian, damping, Broyden."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from newton_nd import (newton, newton_damped, broyden, finite_difference_jacobian)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- circle-line intersection: x^2+y^2=4, y=x -> (sqrt2, sqrt2) -------------
def F_circle(v):
    x, y = v
    return [x * x + y * y - 4, y - x]


def J_circle(v):
    x, y = v
    return [[2 * x, 2 * y], [-1, 1]]


s2 = math.sqrt(2)
root, it, conv = newton(F_circle, [1.0, 3.0], jacobian=J_circle)
check("circle-line converges", conv)
check("circle-line finds (sqrt2, sqrt2)", approx(root[0], s2, 1e-8) and approx(root[1], s2, 1e-8))
check("residual is ~0 at the root", max(abs(c) for c in F_circle(root)) < 1e-8)

# --- finite-difference Jacobian matches the analytic one -------------------
fd = finite_difference_jacobian(F_circle, [1.5, 0.5])
an = J_circle([1.5, 0.5])
check("finite-difference Jacobian matches analytic",
      all(approx(fd[i][j], an[i][j], 1e-4) for i in range(2) for j in range(2)))

# --- Newton without an analytic Jacobian still converges -------------------
root_fd, _, conv_fd = newton(F_circle, [1.0, 3.0])
check("finite-difference Newton converges", conv_fd)
check("finite-difference Newton is accurate", approx(root_fd[0], s2, 1e-6))

# --- Rosenbrock stationary point: grad = 0 at (1, 1) -----------------------
def grad_rosenbrock(v):
    x, y = v
    return [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)]


rroot, rit, rconv = newton(grad_rosenbrock, [-1.0, 2.0])
check("Rosenbrock stationary converges", rconv)
check("Rosenbrock stationary point is (1,1)",
      approx(rroot[0], 1.0, 1e-6) and approx(rroot[1], 1.0, 1e-6))

# --- quadratic convergence near the root -----------------------------------
_, _, _, hist = newton(F_circle, [1.6, 2.4], jacobian=J_circle, track=True)
# once close, each residual is roughly the square of the previous (digits double)
# check the tail shrinks super-linearly: r[k+1] < r[k]^1.5 for the small ones
tail = [h for h in hist if h < 0.5]
quad = all(tail[i + 1] <= tail[i] ** 1.5 + 1e-12 for i in range(len(tail) - 1))
check("convergence is quadratic near the root", quad and hist[-1] < 1e-9)

# --- a 3-variable nonlinear system -----------------------------------------
def F3(v):
    x, y, z = v
    return [x + y + z - 6,          # sums to 6
            x * x + y * y + z * z - 14,   # 1+4+9
            x * y * z - 6]          # 1*2*3
    # a solution is (1, 2, 3)


r3, _, c3 = newton(F3, [0.5, 1.5, 3.5])
check("3-variable system converges", c3)
check("3-variable residual ~0", max(abs(c) for c in F3(r3)) < 1e-7)

# --- damping rescues a start where plain Newton diverges -------------------
# arctan: Newton overshoots and diverges for |x0| large; damped converges to 0
def F_atan(v):
    return [math.atan(v[0])]


def J_atan(v):
    return [[1.0 / (1.0 + v[0] ** 2)]]


plain_diverged = False
try:
    _, _, pconv = newton(F_atan, [5.0], jacobian=J_atan, max_iter=30)
    if not pconv:
        plain_diverged = True
except OverflowError:
    plain_diverged = True
check("plain Newton diverges on arctan from a far start", plain_diverged)
rd, _, cd = newton_damped(F_atan, [5.0], jacobian=J_atan)
check("damped Newton converges where plain diverges", cd and approx(rd[0], 0.0, 1e-6))

# --- damped Newton also solves the ordinary systems ------------------------
rd2, _, cd2 = newton_damped(F_circle, [1.0, 3.0], jacobian=J_circle)
check("damped Newton solves circle-line too", cd2 and approx(rd2[0], s2, 1e-7))

# --- Broyden quasi-Newton converges ----------------------------------------
rb, _, cb = broyden(F_circle, [1.0, 3.0])
check("Broyden converges", cb)
check("Broyden finds the root", approx(rb[0], s2, 1e-6) and approx(rb[1], s2, 1e-6))
rb3, _, cb3 = broyden(F3, [0.5, 1.5, 3.5])
check("Broyden solves the 3-variable system", cb3 and max(abs(c) for c in F3(rb3)) < 1e-6)

# --- starting at the root returns immediately ------------------------------
r0, it0, c0 = newton(F_circle, [s2, s2], jacobian=J_circle)
check("starting at the root converges immediately", c0 and it0 == 1)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all newton_nd tests passed")
