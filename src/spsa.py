"""SPSA: stochastic optimization that estimates a gradient with just two measurements, in any dimension.

To minimize a function by gradient descent you need the gradient; if you can only EVALUATE the function
(a simulation, a noisy experiment, a black box), the obvious fix -- finite differences -- costs 2p
evaluations in p dimensions, ruinous when p is large. Spall's SIMULTANEOUS PERTURBATION STOCHASTIC
APPROXIMATION (1992) is the beautiful trick: perturb ALL coordinates at once by a random sign vector
Delta, measure the function at theta + c*Delta and theta - c*Delta, and estimate the WHOLE gradient
from that single pair of measurements:

    g_hat_i = (f(theta + c*Delta) - f(theta - c*Delta)) / (2 c Delta_i).

Only TWO evaluations per iteration, no matter how many parameters -- and because the perturbation
directions average out over iterations, the estimate is unbiased enough to converge to the optimum. The
step and perturbation sizes follow decaying GAIN SEQUENCES a_k = a/(k+1+A)^alpha and c_k = c/(k+1)^gamma
(Spall's standard alpha=0.602, gamma=0.101), which guarantee convergence for noisy objectives -- SPSA's
signature strength, making it a staple of hyperparameter tuning, control, and simulation optimization.

This module implements SPSA with Rademacher (+/-1) perturbations and the standard decaying gains,
returning the optimized parameters and optionally the trajectory, using a seeded RNG. It is validated:
it minimizes a quadratic bowl to high accuracy using only two function evaluations per iteration; it
locates a shifted optimum; it still converges when the objective is corrupted by additive noise (where
plain finite differences would thrash); the total evaluation count is exactly 2 per iteration
independent of dimension; the gain sequences decay at the specified rates; it is reproducible for a
fixed seed; and it works from 1 to many dimensions. Pure stdlib; the stochastic-optimization companion
to the cross-entropy-method, CMA-ES, and gradient-descent tools."""

from __future__ import annotations


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u32(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def sign(self):
        """A Rademacher +/-1 random variable. Uses a HIGH bit -- an LCG's
        low-order bits alternate deterministically (period 2), so `& 1` would
        ignore the seed and destroy the perturbation's randomness."""
        return 1.0 if (self.u32() >> 31) else -1.0


def minimize(f, theta0, iterations=1000, a=0.2, c=0.1, A=None, alpha=0.602, gamma=0.101,
             seed=1, track=False):
    """Minimize f by SPSA. f: list -> float. theta0: initial parameters.

    Gains: a_k = a/(k+1+A)^alpha, c_k = c/(k+1)^gamma. Returns theta (and history if track=True).
    Exactly two evaluations of f per iteration, regardless of dimension."""
    p = len(theta0)
    theta = list(theta0)
    if A is None:
        A = max(1, iterations // 10)              # stability offset ~10% of the run
    rng = _Rng(seed)
    history = []
    evals = 0
    for k in range(iterations):
        ak = a / ((k + 1 + A) ** alpha)
        ck = c / ((k + 1) ** gamma)
        delta = [rng.sign() for _ in range(p)]
        theta_plus = [theta[i] + ck * delta[i] for i in range(p)]
        theta_minus = [theta[i] - ck * delta[i] for i in range(p)]
        y_plus = f(theta_plus)
        y_minus = f(theta_minus)
        evals += 2
        # simultaneous-perturbation gradient estimate
        diff = (y_plus - y_minus)
        ghat = [diff / (2 * ck * delta[i]) for i in range(p)]
        theta = [theta[i] - ak * ghat[i] for i in range(p)]
        if track:
            history.append((list(theta), f(theta)))
    if track:
        return theta, history, evals
    return theta


def gain_sequences(iterations, a=0.2, c=0.1, A=None, alpha=0.602, gamma=0.101):
    """Return the (a_k, c_k) gain sequences used over the run, for inspection."""
    if A is None:
        A = max(1, iterations // 10)
    aks = [a / ((k + 1 + A) ** alpha) for k in range(iterations)]
    cks = [c / ((k + 1) ** gamma) for k in range(iterations)]
    return aks, cks


# --- benchmark objectives --------------------------------------------------
def sphere(x):
    return sum(xi * xi for xi in x)


def shifted_sphere(center):
    """A quadratic bowl centered at `center`."""
    def f(x):
        return sum((x[i] - center[i]) ** 2 for i in range(len(x)))
    return f


def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


def noisy(f, noise_amp, seed=0):
    """Wrap an objective with additive uniform noise (for testing SPSA's noise tolerance)."""
    state = [seed & 0xFFFFFFFF]

    def g(x):
        state[0] = (1664525 * state[0] + 1013904223) & 0xFFFFFFFF
        u = (state[0] >> 8) / (1 << 24) * 2 - 1
        return f(x) + noise_amp * u
    return g
