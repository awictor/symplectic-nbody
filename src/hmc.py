"""Hamiltonian Monte Carlo: sampling hard distributions by rolling a ball on the log-density.

Markov-chain Monte Carlo draws samples from a distribution known only up to a constant. The simplest
version, random-walk Metropolis, proposes a small random step and accepts or rejects it -- but on
correlated or high-dimensional targets it random-walks agonizingly slowly, taking tiny steps to keep
the acceptance rate up. HAMILTONIAN MONTE CARLO (Duane et al. 1987; Neal 2011) does far better by
borrowing physics. Treat the negative log-density as a POTENTIAL ENERGY U(q) = -log p(q), give the
sample a random MOMENTUM p drawn from a Gaussian (kinetic energy K(p) = p^2/2), and let the pair (q,
p) roll along a trajectory of Hamilton's equations. Because the dynamics conserve the total energy
H = U + K and preserve phase-space volume, following the trajectory for a while and then accepting by
a Metropolis test on the tiny energy error gives PROPOSALS FAR FROM THE START with near-certain
acceptance -- exploring the distribution in long, informed sweeps instead of a timid walk.

The trajectory is integrated by the LEAPFROG (velocity-Verlet) scheme, which is symplectic (it
conserves a nearby energy exactly, so the trajectory doesn't drift) and time-reversible (needed for
detailed balance):

    p += (dt/2) * (-grad U(q)),   q += dt * p,   p += (dt/2) * (-grad U(q)),

repeated L steps. The final energy error, tiny for a symplectic integrator, sets the accept
probability min(1, exp(H_old - H_new)). This module runs HMC from a log-density and its gradient (or
a finite-difference gradient), reports the samples and acceptance rate, and includes random-walk
Metropolis for comparison.

Validated: on a standard normal HMC's sample mean and variance match 0 and 1; on a correlated 2-D
Gaussian it recovers the means, variances, and correlation; the acceptance rate is high (the
symplectic integrator keeps the energy error small); it mixes far better than random-walk Metropolis
on the correlated target (lower autocorrelation at the same cost); and reducing the step size raises
acceptance toward 1. Pure stdlib; the gradient-guided sampler completing the Metropolis / Gibbs MCMC
family."""

from __future__ import annotations

import math


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)

    return nxt


def _gaussian(rng):
    u1 = max(rng(), 1e-12)
    u2 = rng()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def finite_diff_grad(logp, q, h=1e-5):
    """Numerical gradient of logp at q by central differences."""
    n = len(q)
    g = [0.0] * n
    for i in range(n):
        qp = list(q)
        qm = list(q)
        qp[i] += h
        qm[i] -= h
        g[i] = (logp(qp) - logp(qm)) / (2 * h)
    return g


def hmc(logp, q0, n_samples, step_size=0.1, n_leapfrog=20, grad=None, seed=12345, burn_in=0):
    """Hamiltonian Monte Carlo. logp(q) is the (unnormalized) log-density; grad(q) its gradient
    (finite-differenced if None). Returns (samples, acceptance_rate)."""
    rng = _lcg(seed)
    if grad is None:
        def grad(q):
            return finite_diff_grad(logp, q)
    q = list(q0)
    dim = len(q)
    samples = []
    accepts = 0
    total = n_samples + burn_in
    for it in range(total):
        # sample momentum
        p = [_gaussian(rng) for _ in range(dim)]
        current_q = list(q)
        current_p = list(p)
        # leapfrog: U = -logp, so -grad U = grad logp
        g = grad(q)
        # half step for momentum
        p = [p[i] + 0.5 * step_size * g[i] for i in range(dim)]
        prop_q = list(q)
        for step in range(n_leapfrog):
            prop_q = [prop_q[i] + step_size * p[i] for i in range(dim)]
            g = grad(prop_q)
            if step != n_leapfrog - 1:
                p = [p[i] + step_size * g[i] for i in range(dim)]
        # final half step
        p = [p[i] + 0.5 * step_size * g[i] for i in range(dim)]
        # Metropolis on the Hamiltonian H = -logp + 0.5|p|^2
        current_U = -logp(current_q)
        current_K = 0.5 * sum(x * x for x in current_p)
        prop_U = -logp(prop_q)
        prop_K = 0.5 * sum(x * x for x in p)
        dH = (current_U + current_K) - (prop_U + prop_K)
        if rng() < math.exp(min(0.0, dH)):
            q = prop_q
            if it >= burn_in:
                accepts += 1
        if it >= burn_in:
            samples.append(list(q))
    rate = accepts / n_samples if n_samples else 0.0
    return samples, rate


def random_walk_metropolis(logp, q0, n_samples, step_size=0.5, seed=12345, burn_in=0):
    """Random-walk Metropolis for comparison. Returns (samples, acceptance_rate)."""
    rng = _lcg(seed)
    q = list(q0)
    dim = len(q)
    samples = []
    accepts = 0
    cur_lp = logp(q)
    for it in range(n_samples + burn_in):
        prop = [q[i] + step_size * _gaussian(rng) for i in range(dim)]
        prop_lp = logp(prop)
        if rng() < math.exp(min(0.0, prop_lp - cur_lp)):
            q = prop
            cur_lp = prop_lp
            if it >= burn_in:
                accepts += 1
        if it >= burn_in:
            samples.append(list(q))
    return samples, accepts / n_samples if n_samples else 0.0


# --- diagnostics -------------------------------------------------------------
def sample_mean(samples):
    n = len(samples)
    dim = len(samples[0])
    return [sum(s[i] for s in samples) / n for i in range(dim)]


def sample_cov(samples):
    n = len(samples)
    dim = len(samples[0])
    mean = sample_mean(samples)
    cov = [[0.0] * dim for _ in range(dim)]
    for s in samples:
        for i in range(dim):
            for j in range(dim):
                cov[i][j] += (s[i] - mean[i]) * (s[j] - mean[j])
    for i in range(dim):
        for j in range(dim):
            cov[i][j] /= (n - 1)
    return cov


def autocorrelation(samples, dim_index, lag):
    """Lag-`lag` autocorrelation of one coordinate of the chain (a mixing diagnostic)."""
    xs = [s[dim_index] for s in samples]
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    if var == 0:
        return 0.0
    c = sum((xs[i] - mean) * (xs[i + lag] - mean) for i in range(n - lag)) / (n - lag)
    return c / var
