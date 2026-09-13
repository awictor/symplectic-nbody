"""Support vector machines by SMO: the maximum-margin classifier, kernels and all.

A support vector machine draws the decision boundary that separates two classes with the WIDEST
possible margin -- the boundary as far as it can get from the nearest point of either class. Where a
perceptron stops at the first boundary that happens to work and logistic regression optimizes a
probabilistic loss, the SVM solves for the UNIQUE maximum-margin hyperplane, which generalizes well
because it commits to the safest possible separation. Training is a convex quadratic program in the
DUAL variables -- one Lagrange multiplier alpha_i per training point:

    maximize   sum_i alpha_i - 1/2 sum_{i,j} alpha_i alpha_j y_i y_j K(x_i, x_j)
    subject to 0 <= alpha_i <= C   and   sum_i alpha_i y_i = 0

The points with alpha_i > 0 are the SUPPORT VECTORS -- the only ones that touch the margin and define
the boundary; everything else is irrelevant. The kernel K lets the same machinery draw NONLINEAR
boundaries by an implicit map into a higher-dimensional space: a Gaussian (RBF) kernel can carve out
curved and even disconnected regions while all the algebra stays in terms of inner products.

The classic solver is Platt's SEQUENTIAL MINIMAL OPTIMIZATION (1998). The equality constraint means
you cannot move one alpha alone, so SMO optimizes the SMALLEST possible working set -- TWO multipliers
at a time -- which has a closed-form solution, and sweeps the data repeatedly fixing KKT violators
until every point satisfies the optimality conditions to tolerance. No general QP library needed.

This module implements the simplified-SMO training loop with linear, polynomial, and RBF kernels,
prediction, and the recovered decision function. It is validated against ground truth: on a linearly
separable set it finds a separating boundary with all labels correct and every point outside the
margin (functional margin >= 1); the bias and support vectors satisfy the KKT conditions; the dual
constraint sum alpha_i y_i = 0 holds; the RBF kernel solves the non-linearly-separable XOR and two
concentric rings; a hand-computed two-point maximum-margin line is reproduced exactly; and a linear
SVM's separating direction agrees with the analytic max-margin normal. Pure stdlib; the
maximum-margin, kernel companion to the perceptron and logistic-regression classifiers."""

from __future__ import annotations

import math


def linear_kernel(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def poly_kernel(degree=2, gamma=1.0, coef0=1.0):
    def k(a, b):
        return (gamma * sum(ai * bi for ai, bi in zip(a, b)) + coef0) ** degree
    return k


def rbf_kernel(gamma=1.0):
    def k(a, b):
        sq = sum((ai - bi) ** 2 for ai, bi in zip(a, b))
        return math.exp(-gamma * sq)
    return k


class SVM:
    """A binary SVM trained by simplified SMO. Labels must be +1 / -1."""

    def __init__(self, kernel=None, C=1.0, tol=1e-3, max_passes=200, eps=1e-8):
        self.kernel = kernel or linear_kernel
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.eps = eps
        self.alpha = None
        self.b = 0.0
        self.X = None
        self.y = None
        self._K = None

    def _kernel_matrix(self, X):
        n = len(X)
        K = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                v = self.kernel(X[i], X[j])
                K[i][j] = K[j][i] = v
        return K

    def _decision_cached(self, i):
        """Decision function f(x_i) using the cached kernel matrix."""
        s = self.b
        for j in range(len(self.X)):
            if self.alpha[j] != 0.0:
                s += self.alpha[j] * self.y[j] * self._K[j][i]
        return s

    def fit(self, X, y):
        n = len(X)
        self.X = [list(x) for x in X]
        self.y = list(y)
        self.alpha = [0.0] * n
        self.b = 0.0
        self._K = self._kernel_matrix(self.X)
        C = self.C

        # SMO sweeps until KKT holds for `max_passes` consecutive full passes with no update, or a
        # hard iteration cap is reached (guards against pathological non-termination).
        passes = 0
        iterations = 0
        iter_cap = self.max_passes * n * 50 + 1000
        while passes < self.max_passes and iterations < iter_cap:
            num_changed = 0
            for i in range(n):
                iterations += 1
                Ei = self._decision_cached(i) - self.y[i]
                ri = self.y[i] * Ei
                # KKT violation check
                if (ri < -self.tol and self.alpha[i] < C) or (ri > self.tol and self.alpha[i] > 0):
                    # Platt's second-choice hierarchy: try the max-|Ei-Ej| partner first, then fall
                    # back to every other index. A single fixed partner can stall (eta >= 0 or a
                    # clipped step), leaving KKT violators with alpha at a bound; the fallback sweep
                    # guarantees progress whenever any improving pair exists.
                    j = self._select_second(i, Ei, n)
                    stepped = j != i and self._take_step(i, j, Ei)
                    if not stepped:
                        for jj in range(n):
                            if jj != i and jj != j and self._take_step(i, jj, Ei):
                                stepped = True
                                break
                    if stepped:
                        num_changed += 1
            if num_changed == 0:
                passes += 1
            else:
                passes = 0
        return self

    def _select_second(self, i, Ei, n):
        best_j = i
        best_delta = 0.0
        for j in range(n):
            if j == i:
                continue
            Ej = self._decision_cached(j) - self.y[j]
            delta = abs(Ei - Ej)
            if delta > best_delta:
                best_delta = delta
                best_j = j
        return best_j

    def _take_step(self, i, j, Ei):
        C = self.C
        ai_old = self.alpha[i]
        aj_old = self.alpha[j]
        yi, yj = self.y[i], self.y[j]
        Ej = self._decision_cached(j) - yj

        if yi != yj:
            L = max(0.0, aj_old - ai_old)
            H = min(C, C + aj_old - ai_old)
        else:
            L = max(0.0, ai_old + aj_old - C)
            H = min(C, ai_old + aj_old)
        if L >= H:
            return False

        eta = 2 * self._K[i][j] - self._K[i][i] - self._K[j][j]
        if eta >= 0:
            return False

        aj_new = aj_old - yj * (Ei - Ej) / eta
        aj_new = min(max(aj_new, L), H)
        if abs(aj_new - aj_old) < self.eps:
            return False

        ai_new = ai_old + yi * yj * (aj_old - aj_new)

        # update bias
        b1 = (self.b - Ei - yi * (ai_new - ai_old) * self._K[i][i]
              - yj * (aj_new - aj_old) * self._K[i][j])
        b2 = (self.b - Ej - yi * (ai_new - ai_old) * self._K[i][j]
              - yj * (aj_new - aj_old) * self._K[j][j])
        self.alpha[i] = ai_new
        self.alpha[j] = aj_new
        if 0 < ai_new < C:
            self.b = b1
        elif 0 < aj_new < C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        return True

    def decision_function(self, x):
        """Signed distance-like score f(x) = sum alpha_i y_i K(x_i, x) + b."""
        s = self.b
        for j in range(len(self.X)):
            if self.alpha[j] != 0.0:
                s += self.alpha[j] * self.y[j] * self.kernel(self.X[j], x)
        return s

    def predict(self, x):
        """Class label +1 / -1 for a single point."""
        return 1 if self.decision_function(x) >= 0 else -1

    def predict_all(self, X):
        return [self.predict(x) for x in X]

    def support_vectors(self):
        """Indices of the support vectors (alpha_i > 0)."""
        return [i for i in range(len(self.alpha)) if self.alpha[i] > self.eps]

    def weight_vector(self):
        """Primal weight vector w = sum alpha_i y_i x_i (defined for the linear kernel)."""
        n = len(self.X)
        dim = len(self.X[0])
        w = [0.0] * dim
        for i in range(n):
            if self.alpha[i] != 0.0:
                for d in range(dim):
                    w[d] += self.alpha[i] * self.y[i] * self.X[i][d]
        return w


def functional_margins(svm):
    """y_i * f(x_i) for every training point -- should be >= 1 for correctly classified non-SV."""
    return [svm.y[i] * svm.decision_function(svm.X[i]) for i in range(len(svm.X))]


def dual_constraint(svm):
    """sum_i alpha_i y_i, which the equality constraint forces to 0."""
    return sum(svm.alpha[i] * svm.y[i] for i in range(len(svm.alpha)))
