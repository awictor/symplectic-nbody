"""The Ornstein-Uhlenbeck process: the mean-reverting random walk with exact everything.

A drunkard's walk wanders off to infinity; tie the drunkard to a lamppost with a spring and you get the
Ornstein-Uhlenbeck process -- the simplest stochastic process that both fluctuates AND pulls back toward a
mean. It is the continuous-time analogue of an AR(1) series, the velocity of a Brownian particle under
friction (its 1930 origin), the Vasicek model of interest rates, and the canonical noisy relaxation. Its
defining stochastic differential equation is

    dX = theta (mu - X) dt + sigma dW,

a deterministic restoring drift theta(mu - X) toward the long-run mean mu, plus white-noise kicks of
intensity sigma. Unlike most SDEs it is EXACTLY solvable -- Gaussian at all times with closed forms for
every moment, which makes it the perfect instrument to validate a stochastic integrator against truth:

  MEAN.  E[X_t | X_0] = mu + (X_0 - mu) e^{-theta t}  -- exponential relaxation to mu with rate theta.
  VARIANCE.  Var[X_t | X_0] = (sigma^2 / 2 theta)(1 - e^{-2 theta t})  -- grows from 0 to the stationary
      value sigma^2/(2 theta).
  STATIONARY LAW.  As t -> infinity, X ~ Normal(mu, sigma^2/(2 theta)), independent of the start.
  AUTOCOVARIANCE.  In the stationary state, Cov(X_t, X_{t+s}) = (sigma^2/2 theta) e^{-theta |s|}: memory
      decays exponentially with correlation time 1/theta.
  EXACT UPDATE.  X_{t+dt} = mu + (X_t - mu) e^{-theta dt} + sqrt((sigma^2/2 theta)(1 - e^{-2 theta dt})) Z,
      Z standard normal -- a discretization with NO time-step error, valid for any dt.

This module provides the analytic moments, an exact Gaussian stepper, the naive Euler-Maruyama stepper
(for contrast), a stationary sampler, and empirical estimators. It is validated against the closed forms:
the simulated mean and variance follow the exponential-relaxation curves; the stationary histogram matches
Normal(mu, sigma^2/2theta); the empirical autocovariance decays like e^{-theta s}; the exact stepper
reproduces the analytic variance at large dt where Euler-Maruyama is badly biased; and the discrete process
is the AR(1) recursion with the right coefficient e^{-theta dt}. Seeded RNG with Box-Muller normals; pure
stdlib. The mean-reverting companion to the random-walk, Langevin, and Kalman-filter notes."""

from __future__ import annotations

import math


class _Rng:
    """Seeded LCG with cached Box-Muller normal (high bits only; low LCG bits are period-2 correlated)."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF
        self._spare = None

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def normal(self):
        if self._spare is not None:
            z = self._spare
            self._spare = None
            return z
        u1 = max(self.u(), 1e-12)
        u2 = self.u()
        r = math.sqrt(-2.0 * math.log(u1))
        self._spare = r * math.sin(2.0 * math.pi * u2)
        return r * math.cos(2.0 * math.pi * u2)


class OrnsteinUhlenbeck:
    """An OU process dX = theta(mu - X) dt + sigma dW with parameters (theta>0, mu, sigma>0)."""

    def __init__(self, theta=1.0, mu=0.0, sigma=1.0):
        if theta <= 0 or sigma <= 0:
            raise ValueError("require theta > 0 and sigma > 0")
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    # ---- analytic moments ----
    def mean(self, x0, t):
        """E[X_t | X_0=x0] = mu + (x0 - mu) e^{-theta t}."""
        return self.mu + (x0 - self.mu) * math.exp(-self.theta * t)

    def variance(self, t):
        """Var[X_t | X_0] = (sigma^2 / 2 theta)(1 - e^{-2 theta t}); independent of the start."""
        return self.sigma ** 2 / (2.0 * self.theta) * (1.0 - math.exp(-2.0 * self.theta * t))

    def stationary_variance(self):
        """The t -> infinity variance sigma^2/(2 theta)."""
        return self.sigma ** 2 / (2.0 * self.theta)

    def autocovariance(self, s):
        """Stationary Cov(X_t, X_{t+s}) = (sigma^2/2 theta) e^{-theta |s|}."""
        return self.stationary_variance() * math.exp(-self.theta * abs(s))

    def autocorrelation(self, s):
        """Stationary correlation e^{-theta |s|} (autocovariance normalized by the variance)."""
        return math.exp(-self.theta * abs(s))

    def correlation_time(self):
        """The memory timescale 1/theta."""
        return 1.0 / self.theta

    # ---- steppers ----
    def step_exact(self, x, dt, rng):
        """Exact Gaussian update -- no time-step discretization error, valid for any dt.

        X_{t+dt} = mu + (x - mu) e^{-theta dt} + sqrt(Var) Z, with Var = (sigma^2/2theta)(1-e^{-2 theta dt})."""
        decay = math.exp(-self.theta * dt)
        var = self.stationary_variance() * (1.0 - decay * decay)
        return self.mu + (x - self.mu) * decay + math.sqrt(var) * rng.normal()

    def step_euler(self, x, dt, rng):
        """Naive Euler-Maruyama update X + theta(mu - X) dt + sigma sqrt(dt) Z. Biased for large dt."""
        return x + self.theta * (self.mu - x) * dt + self.sigma * math.sqrt(dt) * rng.normal()

    def simulate(self, x0, dt, steps, rng, exact=True):
        """Integrate one path from x0 for `steps` steps of size dt. Returns the list of states (len steps+1)."""
        step = self.step_exact if exact else self.step_euler
        x = x0
        out = [x]
        for _ in range(steps):
            x = step(x, dt, rng)
            out.append(x)
        return out

    def sample_stationary(self, rng):
        """Draw one sample from the stationary law Normal(mu, sigma^2/2theta)."""
        return self.mu + math.sqrt(self.stationary_variance()) * rng.normal()

    # ---- empirical estimators (for validation / demos) ----
    def empirical_moments(self, x0, t, dt, paths, seed=1, exact=True):
        """Monte-Carlo estimate of (E[X_t], Var[X_t]) from `paths` independent trajectories to time t.

        Spread the per-path seeds (x7919) so the LCG streams decorrelate."""
        steps = max(1, int(round(t / dt)))
        vals = []
        for p in range(paths):
            rng = _Rng((seed + p) * 7919 + 1)
            path = self.simulate(x0, dt, steps, rng, exact=exact)
            vals.append(path[-1])
        n = len(vals)
        m = sum(vals) / n
        v = sum((x - m) ** 2 for x in vals) / (n - 1)
        return m, v

    def empirical_autocovariance(self, dt, steps, rng, lags, burn=None):
        """Estimate the stationary autocovariance at the given integer `lags` from one long path.

        Burn in (default steps//5) to reach stationarity, then compute lag products about the sample mean."""
        if burn is None:
            burn = steps // 5
        # start at the stationary mean to shorten burn-in
        path = self.simulate(self.mu, dt, steps, rng, exact=True)[burn:]
        n = len(path)
        m = sum(path) / n
        out = {}
        for lag in lags:
            cov = sum((path[i] - m) * (path[i + lag] - m) for i in range(n - lag)) / (n - lag)
            out[lag] = cov
        return out
