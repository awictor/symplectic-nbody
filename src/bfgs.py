"""BFGS: quasi-Newton optimization that learns curvature from gradients alone.

To minimize a smooth function f, gradient descent steps downhill but crawls through narrow valleys.
Newton's method rescales the step by the inverse HESSIAN (the matrix of second derivatives),
accounting for curvature so it converges SUPERLINEARLY -- but forming and inverting the Hessian needs
second derivatives and O(n^3) work. BFGS (Broyden-Fletcher-Goldfarb-Shanno, 1970) is the celebrated
quasi-Newton method: it maintains an APPROXIMATION to the inverse Hessian, updated at each step from
only the change in position and the change in gradient (the SECANT condition), so it needs no second
derivatives yet still converges superlinearly near the optimum.

The update keeps the approximation H symmetric and positive definite (so the search direction -H g is
always a descent direction) via the rank-2 formula

    H' = (I - rho s y^T) H (I - rho y s^T) + rho s s^T,   rho = 1/(y^T s),

where s = x' - x is the step and y = g' - g the gradient change. Each iteration picks the direction
-H g, finds a step length by a backtracking line search satisfying the Armijo (sufficient-decrease)
condition, and updates H. This module runs BFGS from a function and its gradient (finite-differenced
if not supplied), returning the minimizer, value, and iteration count.

Validated against analytic optima and convergence theory: BFGS finds the minimum of a quadratic in a
few steps and of the Rosenbrock banana valley to high precision; the gradient vanishes at the
solution; the objective decreases monotonically (a descent method); it matches a known minimum for
several test functions; the inverse-Hessian approximation stays positive definite; and it works with a
finite-difference gradient. Pure stdlib; the full-matrix quasi-Newton companion to the limited-memory
L-BFGS and the Nelder-Mead / gradient-free optimizers."""

from __future__ import annotations


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


def _norm(a):
    return _dot(a, a) ** 0.5


def finite_diff_grad(f, x, h=1e-6):
    """Central-difference gradient of f at x."""
    n = len(x)
    g = [0.0] * n
    for i in range(n):
        xp = list(x)
        xm = list(x)
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


def _matvec(H, v):
    return [sum(H[i][j] * v[j] for j in range(len(v))) for i in range(len(H))]


def _line_search(f, x, fx, g, direction, c1=1e-4, shrink=0.5, max_iter=50):
    """Backtracking line search satisfying the Armijo sufficient-decrease condition."""
    slope = _dot(g, direction)
    if slope >= 0:
        # not a descent direction; fall back to steepest descent
        direction = [-gi for gi in g]
        slope = _dot(g, direction)
    alpha = 1.0
    for _ in range(max_iter):
        x_new = [x[i] + alpha * direction[i] for i in range(len(x))]
        if f(x_new) <= fx + c1 * alpha * slope:
            return alpha, x_new
        alpha *= shrink
    return alpha, [x[i] + alpha * direction[i] for i in range(len(x))]


def minimize(f, x0, grad=None, tol=1e-8, max_iter=1000, track=False):
    """Minimize f starting from x0 by BFGS. grad(x) is the gradient (finite-differenced if None).
    Returns a dict with the minimizer x, value fun, gradient-norm, and iterations."""
    n = len(x0)
    if grad is None:
        def grad(x):
            return finite_diff_grad(f, x)
    x = list(x0)
    g = grad(x)
    fx = f(x)
    # initial inverse-Hessian approximation = identity
    H = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    history = [fx] if track else None
    it = 0
    for it in range(1, max_iter + 1):
        gnorm = _norm(g)
        if gnorm < tol:
            break
        # search direction p = -H g
        p = [-v for v in _matvec(H, g)]
        alpha, x_new = _line_search(f, x, fx, g, p)
        g_new = grad(x_new)
        s = [x_new[i] - x[i] for i in range(n)]
        y = [g_new[i] - g[i] for i in range(n)]
        sy = _dot(s, y)
        if sy > 1e-12:
            rho = 1.0 / sy
            # H' = (I - rho s y^T) H (I - rho y s^T) + rho s s^T
            # compute Hy = H y
            Hy = _matvec(H, y)
            yHy = _dot(y, Hy)
            new_H = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    new_H[i][j] = (H[i][j]
                                   - rho * (s[i] * Hy[j] + Hy[i] * s[j])
                                   + rho * rho * yHy * s[i] * s[j]
                                   + rho * s[i] * s[j])
            H = new_H
        x = x_new
        g = g_new
        fx = f(x)
        if track:
            history.append(fx)
    result = {"x": x, "fun": fx, "grad_norm": _norm(g), "iterations": it}
    if track:
        result["history"] = history
    return result
