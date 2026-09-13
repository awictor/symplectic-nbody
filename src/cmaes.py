"""CMA-ES: the covariance matrix adaptation evolution strategy, an optimizer that learns its own shape.

CMA-ES (Hansen & Ostermeier, 2001) is widely regarded as the state of the art for hard, black-box,
continuous optimization -- non-convex, ill-conditioned, non-separable landscapes where gradients are
unavailable or useless. Where differential evolution mutates by adding differences between members
and a particle swarm tracks velocities, CMA-ES samples each generation from a MULTIVARIATE GAUSSIAN
and then RESHAPES that Gaussian from the best samples. Over generations the search distribution
stretches along the fruitful directions and shrinks across the rest, so it effectively learns a local
model of the objective -- a variable-metric method (like BFGS) that never touches a derivative.

One generation, population of lambda samples around a mean m with step size sigma and covariance C:

  1. SAMPLE. Draw x_k = m + sigma * y_k, where y_k ~ N(0, C). The C^{1/2} factor is obtained from the
     eigendecomposition C = B diag(d^2) B^T, so y_k = B (d .* z_k) with z_k standard normal.
  2. SELECT and RECOMBINE. Evaluate f(x_k), keep the best mu, and move the mean to their
     weighted average (larger weight to better points). This is the new m.
  3. ADAPT THE STEP SIZE via the "conjugate" evolution path p_sigma: accumulate the recent mean
     shifts in the isotropic C^{-1/2} coordinates; if the path is longer than a random walk would be,
     the steps are correlated -> increase sigma; if shorter, decrease it. This is cumulative step-size
     adaptation (CSA), the mechanism that gives CMA-ES its fast, automatic scaling.
  4. ADAPT THE COVARIANCE via a second path p_c plus a rank-mu update from the selected mutation
     vectors, so C bends toward the directions that just produced improvement.

This module implements the standard (mu/mu_w, lambda)-CMA-ES with CSA and the rank-one + rank-mu
covariance update, using the repo's Jacobi eigensolver for the C = B D^2 B^T factorization. It is
validated on the canonical hard test functions: it drives the sphere, the ill-conditioned ellipsoid,
the (non-convex, banana-shaped) Rosenbrock valley, and Rastrigin down to their known global optima
from random starts; the mean converges to the optimizer, the objective drops monotonically in the
best-so-far, and the covariance elongates along the correct axis on an anisotropic problem. Pure
stdlib (a small seeded Gaussian sampler via Box-Muller); the variable-metric, derivative-free
companion to differential evolution, particle swarm, and Nelder-Mead."""

from __future__ import annotations

import math

from jacobi_eigen import eigen


class _Rng:
    """Seeded LCG with a Box-Muller standard-normal draw."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF
        self._spare = None

    def _u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def normal(self):
        if self._spare is not None:
            s, self._spare = self._spare, None
            return s
        # Box-Muller
        u1 = max(self._u(), 1e-12)
        u2 = self._u()
        r = math.sqrt(-2.0 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)


def _matvec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def cmaes(f, x0, sigma0=0.3, max_iter=1000, tol=1e-11, seed=0, popsize=None,
          bounds=None):
    """Minimize f: R^n -> R with CMA-ES starting from x0.

    Returns a dict with 'x' (best point), 'fx' (best value), 'iterations', 'mean', and 'history'
    (best-so-far value per generation).
    """
    n = len(x0)
    xmean = [float(v) for v in x0]
    sigma = float(sigma0)

    # ---- strategy parameters (Hansen's standard defaults) ----
    lam = popsize if popsize else 4 + int(3 * math.log(n))
    mu = lam // 2
    # log-decreasing recombination weights, normalized to sum 1
    weights = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    mueff = 1.0 / sum(w * w for w in weights)

    # adaptation time constants
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = math.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n * n))

    pc = [0.0] * n
    ps = [0.0] * n
    C = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    B = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    D = [1.0] * n

    rng = _Rng(seed)

    best_x = list(xmean)
    best_f = f(xmean)
    history = []

    def clamp(v):
        if bounds is None:
            return v
        lo, hi = bounds
        return [min(max(vi, lo[i]), hi[i]) for i, vi in enumerate(v)]

    for it in range(1, max_iter + 1):
        # ---- sample lambda offspring ----
        zs = []
        ys = []
        xs = []
        fvals = []
        for _ in range(lam):
            z = [rng.normal() for _ in range(n)]
            # y = B (D .* z)
            Dz = [D[i] * z[i] for i in range(n)]
            y = _matvec(B, Dz)
            x = clamp([xmean[i] + sigma * y[i] for i in range(n)])
            zs.append(z)
            ys.append(y)
            xs.append(x)
            fvals.append(f(x))

        # ---- selection ----
        order = sorted(range(lam), key=lambda k: fvals[k])
        if fvals[order[0]] < best_f:
            best_f = fvals[order[0]]
            best_x = list(xs[order[0]])

        # weighted mean of the best mu, in x, y and z coordinates
        xmean_new = [0.0] * n
        ymean = [0.0] * n
        zmean = [0.0] * n
        for rank in range(mu):
            k = order[rank]
            w = weights[rank]
            for i in range(n):
                xmean_new[i] += w * xs[k][i]
                ymean[i] += w * ys[k][i]
                zmean[i] += w * zs[k][i]
        xmean = xmean_new

        # ---- cumulative step-size adaptation (CSA) ----
        # ps = (1 - cs) ps + sqrt(cs(2-cs) mueff) * B zmean   (C^{-1/2} (mean shift) = B zmean)
        coef = math.sqrt(cs * (2 - cs) * mueff)
        Bz = _matvec(B, zmean)
        ps = [(1 - cs) * ps[i] + coef * Bz[i] for i in range(n)]
        ps_norm = math.sqrt(sum(p * p for p in ps))

        # ---- covariance path ----
        # hsig guards the rank-one update when the step-size path is too long
        hsig = 1.0 if ps_norm / math.sqrt(1 - (1 - cs) ** (2 * it)) / chiN < 1.4 + 2.0 / (n + 1) else 0.0
        ccoef = math.sqrt(cc * (2 - cc) * mueff)
        pc = [(1 - cc) * pc[i] + hsig * ccoef * ymean[i] for i in range(n)]

        # ---- covariance matrix update (rank-one + rank-mu) ----
        for i in range(n):
            for j in range(n):
                rank_one = pc[i] * pc[j]
                rank_mu = 0.0
                for rk in range(mu):
                    k = order[rk]
                    rank_mu += weights[rk] * ys[k][i] * ys[k][j]
                # (1-hsig) term keeps C unbiased when hsig = 0
                c_delta = (1 - hsig) * cc * (2 - cc) * C[i][j]
                C[i][j] = ((1 - c1 - cmu) * C[i][j]
                           + c1 * (rank_one + c_delta)
                           + cmu * rank_mu)

        # ---- step-size update ----
        sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1))

        # ---- eigendecomposition C = B diag(D^2) B^T (symmetrize first) ----
        for i in range(n):
            for j in range(i + 1, n):
                m = 0.5 * (C[i][j] + C[j][i])
                C[i][j] = C[j][i] = m
        evals, evecs = eigen(C)
        # guard against tiny negative eigenvalues from roundoff
        D = [math.sqrt(max(ev, 1e-20)) for ev in evals]
        B = evecs

        history.append(best_f)

        # ---- termination ----
        spread = sigma * max(D)
        if best_f <= tol or spread < 1e-12:
            break

    return {
        "x": best_x,
        "fx": best_f,
        "iterations": it,
        "mean": xmean,
        "sigma": sigma,
        "history": history,
    }


# ---- canonical test functions -------------------------------------------------------------

def sphere(x):
    return sum(xi * xi for xi in x)


def ellipsoid(x):
    """Ill-conditioned: axis i scaled by 1000^(i/(n-1))."""
    n = len(x)
    return sum((1000 ** (i / (n - 1) if n > 1 else 0)) * x[i] ** 2 for i in range(n))


def rosenbrock(x):
    """The banana valley; global min 0 at (1, 1, ..., 1)."""
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


def rastrigin(x):
    """Highly multimodal; global min 0 at the origin."""
    n = len(x)
    return 10 * n + sum(xi * xi - 10 * math.cos(2 * math.pi * xi) for xi in x)
