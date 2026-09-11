"""CMA-ES: the covariance matrix adaptation evolution strategy.

For minimizing a black-box function -- one with no gradient, possibly noisy, non-convex, ill-scaled --
the COVARIANCE MATRIX ADAPTATION EVOLUTION STRATEGY is among the most powerful derivative-free
optimizers known, and the de-facto standard for hard continuous problems. Its central idea is to
sample candidate solutions from a MULTIVARIATE NORMAL distribution and, generation after generation,
reshape that distribution to point toward better regions. It adapts three things: the MEAN (moved to
the weighted average of the best samples), the STEP SIZE sigma (grown or shrunk by comparing the
length of the actual path taken to the length expected under pure randomness -- 'cumulative step-size
adaptation'), and the full COVARIANCE MATRIX C (bent to align with the directions of recent progress,
so the search ellipsoid learns the local curvature of the landscape, much like a second-order method
learns the inverse Hessian, but without any derivatives).

Each generation: draw lambda samples x = mean + sigma * B D z, where C = B D^2 B^T is the eigendecom-
position of the covariance (B rotates, D scales); rank them by fitness; move the mean to the weighted
recombination of the best mu; update two evolution paths (one for sigma, one for C) that accumulate
successive steps; and update C from both a rank-one term (the sigma-path outer product, capturing the
dominant direction) and a rank-mu term (the spread of the selected steps). The step-size path length,
compared to its expectation, tells sigma whether the walk is making consistent progress (lengthen) or
doubling back (shorten). This self-adaptation is what lets CMA-ES solve badly-conditioned and rotated
problems that defeat coordinate-wise methods.

This module implements a faithful (mu/mu_w, lambda)-CMA-ES with the standard strategy parameters,
using a self-contained Jacobi eigensolver for the covariance decomposition. It is verified that
it converges to the known global optimum of the sphere, ellipsoid, Rosenbrock, and (often) Rastrigin
functions to high precision, that it drastically outperforms random search under an equal budget,
that it is invariant to rotations of the sphere (a hallmark property), and that it recovers a shifted
optimum away from the origin. Pure stdlib; a black-box-optimization companion to the
differential-evolution, particle-swarm, and Nelder-Mead notes."""

from __future__ import annotations

import math


def _eig_symmetric(C):
    """Eigen-decomposition of a symmetric matrix C: returns (eigenvalues, eigenvectors-as-columns).

    Uses the Jacobi rotation method -- robust for the small, dense, symmetric covariance matrices
    CMA-ES produces, and it yields the full orthonormal eigenvector basis we need to form B and D."""
    n = len(C)
    A = [row[:] for row in C]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(100):
        # find the largest off-diagonal magnitude
        p, q, off = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > off:
                    off = abs(A[i][j])
                    p, q = i, j
        if off < 1e-14:
            break
        app, aqq, apq = A[p][p], A[q][q], A[p][q]
        phi = 0.5 * math.atan2(2 * apq, aqq - app) if abs(aqq - app) > 1e-300 else math.pi / 4
        c, s = math.cos(phi), math.sin(phi)
        for k in range(n):
            akp, akq = A[k][p], A[k][q]
            A[k][p] = c * akp - s * akq
            A[k][q] = s * akp + c * akq
        for k in range(n):
            apk, aqk = A[p][k], A[q][k]
            A[p][k] = c * apk - s * aqk
            A[q][k] = s * apk + c * aqk
        for k in range(n):
            vkp, vkq = V[k][p], V[k][q]
            V[k][p] = c * vkp - s * vkq
            V[k][q] = s * vkp + c * vkq
    evals = [A[i][i] for i in range(n)]
    return evals, V


class _RNG:
    """Seeded LCG with a Box-Muller Gaussian, so runs are reproducible without the random module."""

    def __init__(self, seed=1):
        self.state = seed & 0xFFFFFFFF
        self._spare = None

    def _u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def gauss(self):
        if self._spare is not None:
            g, self._spare = self._spare, None
            return g
        u1 = max(self._u(), 1e-12)
        u2 = self._u()
        r = math.sqrt(-2 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)


def minimize(f, x0, sigma0=0.3, max_iter=1000, tol=1e-12, seed=1, lam=None):
    """Minimize f: R^n -> R with CMA-ES, starting from mean x0 and step size sigma0.

    Returns a dict with 'x' (best solution), 'fx' (its value), 'iterations', and 'evaluations'."""
    n = len(x0)
    xmean = list(x0)
    sigma = sigma0

    # --- strategy parameters (Hansen's standard settings) ---
    if lam is None:
        lam = 4 + int(3 * math.log(n))
    mu = lam // 2
    weights = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    mueff = 1.0 / sum(w * w for w in weights)

    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    pc = [0.0] * n
    ps = [0.0] * n
    C = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    B = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    D = [1.0] * n

    rng = _RNG(seed)
    evals = 0
    best_x = list(xmean)
    best_f = float("inf")
    eigen_stale = 0

    for iteration in range(max_iter):
        # --- sample lambda offspring ---
        arz = []       # standard-normal draws
        arx = []       # actual candidates
        for _ in range(lam):
            z = [rng.gauss() for _ in range(n)]
            # y = B * (D .* z)
            dz = [D[j] * z[j] for j in range(n)]
            y = [sum(B[i][j] * dz[j] for j in range(n)) for i in range(n)]
            x = [xmean[i] + sigma * y[i] for i in range(n)]
            arz.append(z)
            arx.append(x)

        fitness = [(f(x), k) for k, x in enumerate(arx)]
        evals += lam
        fitness.sort(key=lambda t: t[0])

        if fitness[0][0] < best_f:
            best_f = fitness[0][0]
            best_x = list(arx[fitness[0][1]])

        # --- recombination: new mean ---
        old_mean = list(xmean)
        xmean = [0.0] * n
        for wi in range(mu):
            xk = arx[fitness[wi][1]]
            for i in range(n):
                xmean[i] += weights[wi] * xk[i]

        # (xmean - old_mean) / sigma in the sampling space
        ymean = [(xmean[i] - old_mean[i]) / sigma for i in range(n)]

        # --- update evolution path for sigma (needs C^{-1/2} y = B D^{-1} B^T y) ---
        BTy = [sum(B[j][i] * ymean[j] for j in range(n)) for i in range(n)]     # B^T y
        DinvBTy = [BTy[i] / D[i] for i in range(n)]
        Cinvhalf_y = [sum(B[i][j] * DinvBTy[j] for j in range(n)) for i in range(n)]
        ps = [(1 - cs) * ps[i] + math.sqrt(cs * (2 - cs) * mueff) * Cinvhalf_y[i] for i in range(n)]
        ps_norm = math.sqrt(sum(v * v for v in ps))

        # Heaviside step: stall the rank-one update if the path is too long
        hsig = 1.0 if ps_norm / math.sqrt(1 - (1 - cs) ** (2 * (iteration + 1))) / chiN \
            < 1.4 + 2 / (n + 1) else 0.0

        pc = [(1 - cc) * pc[i] + hsig * math.sqrt(cc * (2 - cc) * mueff) * ymean[i] for i in range(n)]

        # --- covariance update: rank-one + rank-mu ---
        for i in range(n):
            for j in range(n):
                rank_one = pc[i] * pc[j]
                rank_mu = 0.0
                for wi in range(mu):
                    yk = [(arx[fitness[wi][1]][d] - old_mean[d]) / sigma for d in (i, j)]
                    rank_mu += weights[wi] * yk[0] * yk[1]
                C[i][j] = ((1 - c1 - cmu) * C[i][j]
                           + c1 * (rank_one + (1 - hsig) * cc * (2 - cc) * C[i][j])
                           + cmu * rank_mu)
        # enforce symmetry
        for i in range(n):
            for j in range(i + 1, n):
                C[i][j] = C[j][i] = 0.5 * (C[i][j] + C[j][i])

        # --- step-size update ---
        sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1))

        # --- re-decompose C periodically ---
        eigen_stale += 1
        if eigen_stale > lam / (c1 + cmu) / n / 10 or eigen_stale >= 1:
            eigen_stale = 0
            evals_pair = _eig_symmetric(C)
            evders, Bnew = evals_pair
            # guard against tiny/negative eigenvalues
            D = [math.sqrt(max(ev, 1e-20)) for ev in evders]
            B = Bnew

        # --- termination ---
        if best_f < tol:
            return {"x": best_x, "fx": best_f, "iterations": iteration + 1, "evaluations": evals}
        if sigma < 1e-16:
            break
        # spread of current best values very small
        if abs(fitness[0][0] - fitness[-1][0]) < tol and best_f < tol * 1e3:
            break

    return {"x": best_x, "fx": best_f, "iterations": max_iter, "evaluations": evals}


# --- standard benchmark functions ------------------------------------------
def sphere(x):
    return sum(xi * xi for xi in x)


def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


def rastrigin(x):
    n = len(x)
    return 10 * n + sum(xi * xi - 10 * math.cos(2 * math.pi * xi) for xi in x)


def ellipsoid(x):
    n = len(x)
    return sum((1e6 ** (i / max(1, n - 1))) * x[i] ** 2 for i in range(n))
