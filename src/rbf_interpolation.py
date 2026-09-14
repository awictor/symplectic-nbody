"""Radial basis function interpolation: smooth surfaces through scattered data in any dimension.

Given values sampled at arbitrary, unstructured points -- no grid, any dimension -- how do you build a
smooth function that passes through all of them? RADIAL BASIS FUNCTION interpolation answers with a
strikingly simple ansatz: the interpolant is a weighted sum of one basis function per data point, each
depending only on the DISTANCE to that point,

    s(x) = sum_i w_i phi(||x - x_i||)   (+ an optional low-order polynomial).

Because phi is radial, the method is oblivious to dimension and to how the points are arranged -- it
works the same for 12 points on a line or 500 points scattered across a 5-D cube. The weights are found
by demanding that s interpolate exactly: s(x_j) = f_j for every j. That is a single linear system
Phi w = f, where Phi_{ij} = phi(||x_i - x_j||). For the classic kernels this matrix is invertible for
distinct points, so the interpolant exists and is unique.

The kernel choice sets the character. GAUSSIAN and INVERSE-MULTIQUADRIC are smooth and localized;
MULTIQUADRIC sqrt(r^2 + c^2) is the old reliable of scattered-data fitting; the THIN-PLATE SPLINE
r^2 log r minimizes bending energy (it is literally the shape a thin metal sheet takes when pinned at
the data heights) but its matrix is only conditionally positive-definite, so it needs a polynomial term
and matching side-conditions to be well-posed. That polynomial augmentation also gives POLYNOMIAL
REPRODUCTION: a thin-plate interpolant reproduces any linear (affine) function exactly.

This module builds RBF interpolants for gaussian, multiquadric, inverse-multiquadric, and thin-plate
kernels, with optional linear-polynomial augmentation, solving the interpolation system with the repo's
LU solver. It is validated: the interpolant reproduces its data values at the nodes to machine
precision for every kernel; it recovers smooth test functions (a Gaussian bump, a linear ramp) between
the nodes; the thin-plate spline with polynomial augmentation reproduces linear functions EXACTLY
everywhere; it works unchanged in 1-D, 2-D, and 3-D; and refining the sample density drives the
interpolation error to zero. Pure stdlib; the scattered-data companion to the Gaussian-process,
kriging, and spline tools."""

from __future__ import annotations

import math

from linsolve import solve


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def _phi(r, kind, eps):
    if kind == "gaussian":
        return math.exp(-(eps * r) ** 2)
    if kind == "multiquadric":
        return math.sqrt(1.0 + (eps * r) ** 2)
    if kind == "inverse_multiquadric":
        return 1.0 / math.sqrt(1.0 + (eps * r) ** 2)
    if kind == "thin_plate":
        # r^2 log r, with the removable singularity at r=0 defined as 0
        return 0.0 if r == 0 else r * r * math.log(r)
    if kind == "linear":
        return r
    if kind == "cubic":
        return r ** 3
    raise ValueError(f"unknown kernel {kind}")


class RBFInterpolator:
    """Radial basis function interpolant through (points, values)."""

    def __init__(self, points, values, kind="multiquadric", epsilon=1.0, polynomial=None):
        self.points = [list(p) for p in points]
        self.values = list(values)
        self.kind = kind
        self.eps = epsilon
        self.dim = len(self.points[0])
        n = len(self.points)
        # thin_plate / linear / cubic are conditionally PD -> default to a linear polynomial term
        if polynomial is None:
            polynomial = kind in ("thin_plate", "linear", "cubic")
        self.polynomial = polynomial
        self.pdeg = self.dim + 1 if polynomial else 0     # constant + one per coordinate

        # assemble the (n+pdeg) x (n+pdeg) system:  [Phi  P] [w]   [f]
        #                                           [P^T  0] [c] = [0]
        m = n + self.pdeg
        A = [[0.0] * m for _ in range(m)]
        b = [0.0] * m
        for i in range(n):
            for j in range(n):
                A[i][j] = _phi(_dist(self.points[i], self.points[j]), kind, epsilon)
            b[i] = self.values[i]
        if polynomial:
            for i in range(n):
                # polynomial basis: [1, x1, x2, ...]
                A[i][n] = 1.0
                for d in range(self.dim):
                    A[i][n + 1 + d] = self.points[i][d]
                # transpose block
                A[n][i] = 1.0
                for d in range(self.dim):
                    A[n + 1 + d][i] = self.points[i][d]
        sol = solve(A, b)
        self.weights = sol[:n]
        self.poly_coef = sol[n:] if polynomial else []

    def __call__(self, x):
        return self.evaluate(x)

    def evaluate(self, x):
        x = list(x)
        s = 0.0
        for i, p in enumerate(self.points):
            s += self.weights[i] * _phi(_dist(x, p), self.kind, self.eps)
        if self.polynomial:
            s += self.poly_coef[0]
            for d in range(self.dim):
                s += self.poly_coef[1 + d] * x[d]
        return s


def interpolate(points, values, query, kind="multiquadric", epsilon=1.0, polynomial=None):
    """Convenience: build an interpolant and evaluate it at a list of query points."""
    rbf = RBFInterpolator(points, values, kind, epsilon, polynomial)
    return [rbf.evaluate(q) for q in query]
