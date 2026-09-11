"""L-BFGS: limited-memory quasi-Newton optimization for smooth functions.

When a function is smooth and you can compute its GRADIENT, the fastest general-purpose minimizers are
QUASI-NEWTON methods. Newton's method rescales the gradient by the inverse HESSIAN (the matrix of
second derivatives), taking curvature into account so it converges superlinearly -- but forming and
inverting an n x n Hessian costs O(n^2) memory and O(n^3) time, hopeless for large n. BFGS instead
builds an approximation to the inverse Hessian from successive gradient differences, needing no second
derivatives; L-BFGS ('limited-memory' BFGS) goes further and never stores the matrix at all. It keeps
only the last m pairs of (step, gradient-change) vectors and reconstructs the action of the inverse
Hessian on the gradient through the elegant TWO-LOOP RECURSION -- so the memory is O(m n) and each
iteration is O(m n). This is the workhorse behind training logistic regression and conditional random
fields, large-scale maximum-likelihood, and countless scientific optimizations.

The two-loop recursion is the heart of it. Given the recent history of s_k = x_{k+1} - x_k and
y_k = grad_{k+1} - grad_k, the search direction -H grad is computed by walking backward through the
history subtracting scaled contributions (first loop), scaling by an estimate of the Hessian's
magnitude, then walking forward adding corrections (second loop). A LINE SEARCH satisfying the Wolfe
conditions then picks a step length that decreases the function enough (Armijo) without overshooting
(curvature), which also keeps the curvature pairs positive-definite so the approximation stays valid.

This module implements L-BFGS with the two-loop recursion and a backtracking/Wolfe line search,
taking a function and its gradient (or using an automatic finite-difference gradient). It is verified
against problems with known optima -- the quadratic bowl (one step to the exact minimum in the
full-memory limit), Rosenbrock, ill-conditioned quadratics, and a logistic-regression negative
log-likelihood -- checking that it reaches high-precision optima, that its finite-difference gradient
matches analytic gradients, that it converges far faster than plain gradient descent on
ill-conditioned problems, and that the recovered logistic-regression weights separate the data. Pure
stdlib; a gradient-based-optimization companion to the CMA-ES, conjugate-gradient, and Newton notes."""

from __future__ import annotations

import math


def finite_diff_gradient(f, x, h=1e-7):
    """Central-difference gradient of f at x."""
    n = len(x)
    g = [0.0] * n
    for i in range(n):
        xp = list(x); xp[i] += h
        xm = list(x); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


def _dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def _axpy(a, x, y):
    """a*x + y."""
    return [a * xi + yi for xi, yi in zip(x, y)]


def _norm(x):
    return math.sqrt(sum(xi * xi for xi in x))


def _line_search(f, grad, x, fx, gx, d, c1=1e-4, c2=0.9, max_ls=50):
    """Backtracking line search satisfying the (strong) Wolfe conditions along direction d.

    Returns (alpha, x_new, f_new, g_new) or None if no acceptable step is found."""
    gd = _dot(gx, d)
    if gd >= 0:
        return None                     # not a descent direction
    alpha = 1.0
    lo, hi = 0.0, None
    for _ in range(max_ls):
        x_new = _axpy(alpha, d, x)
        f_new = f(x_new)
        # Armijo (sufficient decrease)
        if f_new > fx + c1 * alpha * gd:
            hi = alpha
            alpha = 0.5 * (lo + hi)
            continue
        g_new = grad(x_new)
        gd_new = _dot(g_new, d)
        # strong curvature condition
        if abs(gd_new) <= c2 * abs(gd):
            return alpha, x_new, f_new, g_new
        if gd_new >= 0:
            hi = alpha
        else:
            lo = alpha
        if hi is None:
            alpha *= 2.0
        else:
            alpha = 0.5 * (lo + hi)
    # fall back to the last Armijo-satisfying point if we have one
    x_new = _axpy(alpha, d, x)
    f_new = f(x_new)
    if f_new < fx:
        return alpha, x_new, f_new, grad(x_new)
    return None


def minimize(f, x0, grad=None, m=10, max_iter=1000, gtol=1e-8, ftol=1e-12):
    """Minimize f: R^n -> R with L-BFGS.

    f: objective. x0: starting point. grad: gradient function (finite differences if None).
    m: history size. Returns a dict with 'x', 'fx', 'iterations', 'converged', 'grad_norm'."""
    if grad is None:
        grad = lambda x: finite_diff_gradient(f, x)   # noqa: E731

    x = list(x0)
    fx = f(x)
    gx = grad(x)

    s_hist = []      # x_{k+1} - x_k
    y_hist = []      # grad_{k+1} - grad_k
    rho_hist = []    # 1 / (y . s)

    iterations = 0
    converged = False
    for iterations in range(1, max_iter + 1):
        gnorm = _norm(gx)
        if gnorm < gtol:
            converged = True
            break

        # --- two-loop recursion to compute the search direction d = -H grad ---
        q = list(gx)
        alphas = []
        for i in range(len(s_hist) - 1, -1, -1):
            a = rho_hist[i] * _dot(s_hist[i], q)
            alphas.append(a)
            q = _axpy(-a, y_hist[i], q)
        # initial Hessian scaling gamma = (s . y) / (y . y)
        if y_hist:
            gamma = _dot(s_hist[-1], y_hist[-1]) / _dot(y_hist[-1], y_hist[-1])
        else:
            gamma = 1.0 / max(gnorm, 1e-8)
        r = [gamma * qi for qi in q]
        alphas.reverse()
        for i in range(len(s_hist)):
            b = rho_hist[i] * _dot(y_hist[i], r)
            r = _axpy(alphas[i] - b, s_hist[i], r)
        d = [-ri for ri in r]

        ls = _line_search(f, grad, x, fx, gx, d)
        if ls is None:
            # reset to steepest descent once; if that also fails, stop
            d = [-gi for gi in gx]
            ls = _line_search(f, grad, x, fx, gx, d)
            if ls is None:
                break
        alpha, x_new, f_new, g_new = ls

        s = [x_new[i] - x[i] for i in range(len(x))]
        y = [g_new[i] - gx[i] for i in range(len(x))]
        sy = _dot(s, y)
        if sy > 1e-12:                  # keep only curvature-positive pairs
            s_hist.append(s)
            y_hist.append(y)
            rho_hist.append(1.0 / sy)
            if len(s_hist) > m:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)

        if abs(fx - f_new) < ftol * max(1.0, abs(fx)):
            x, fx, gx = x_new, f_new, g_new
            converged = True
            break
        x, fx, gx = x_new, f_new, g_new

    return {"x": x, "fx": fx, "iterations": iterations, "converged": converged,
            "grad_norm": _norm(gx)}
