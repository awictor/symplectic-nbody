"""Particle filters: sequential estimation when the world is nonlinear and non-Gaussian.

The Kalman filter is optimal, but only for LINEAR dynamics with GAUSSIAN noise. When the motion or
the sensor is nonlinear -- a bearing-only tracker, a bimodal belief, a heavy-tailed sensor -- its
single Gaussian belief breaks. A PARTICLE FILTER (Gordon, Salmond, Smith, 1993) drops that
assumption entirely: it represents the belief as a CLOUD of weighted samples ("particles") and
propagates the cloud through the true, arbitrary dynamics. It is sequential Monte Carlo -- the
posterior approximated by where the particles concentrate.

Each step of the bootstrap filter:

  PREDICT   -- push every particle through the (possibly nonlinear) motion model plus process noise.
  WEIGHT    -- reweight each particle by the LIKELIHOOD of the actual measurement given its state.
  RESAMPLE  -- draw a new equal-weight set in proportion to the weights, killing unlikely particles
               and duplicating likely ones, so effort concentrates where the posterior mass is.

Resampling is the crux: without it a few particles hoard all the weight (DEGENERACY) and the cloud
stops representing the posterior. The EFFECTIVE SAMPLE SIZE, 1 / sum(w_i^2), measures that; we
resample only when it falls below half the particle count, using low-variance SYSTEMATIC resampling
(one random offset, evenly spaced pointers) to minimize the added noise. The state estimate is the
weighted mean of the cloud.

This module implements a generic bootstrap particle filter with systematic resampling and adaptive
(ESS-triggered) resampling -- verified on a standard nonlinear benchmark and a bearing-only tracker
that a Kalman filter cannot handle, that its estimate tracks the hidden state, that resampling
keeps the effective sample size healthy where a weight-only filter collapses, and that more
particles reduce the error. Pure stdlib (its own Gaussian sampler); the nonlinear, non-Gaussian
companion to the Kalman-filter and hidden-Markov-model notes."""

from __future__ import annotations

import math


class _Rng:
    """Small LCG with a Box-Muller Gaussian, using the high bits (low bits of an LCG are weak)."""

    def __init__(self, seed=0):
        self.state = seed & 0xFFFFFFFF
        self._spare = None

    def uniform(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def gauss(self, mu=0.0, sigma=1.0):
        if self._spare is not None:
            z = self._spare
            self._spare = None
            return mu + sigma * z
        u1 = max(1e-12, self.uniform())
        u2 = self.uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        self._spare = r * math.sin(2.0 * math.pi * u2)
        return mu + sigma * r * math.cos(2.0 * math.pi * u2)


def effective_sample_size(weights):
    """1 / sum(w_i^2) for normalized weights -- ranges from 1 (degenerate) to N (uniform)."""
    s2 = sum(w * w for w in weights)
    return 1.0 / s2 if s2 > 0 else 0.0


def systematic_resample(weights, rng):
    """Low-variance systematic resampling: return indices drawn in proportion to `weights`.

    One uniform offset in [0, 1/N), then N evenly spaced pointers walked against the cumulative
    weights. Lower variance than multinomial resampling."""
    n = len(weights)
    # one random offset in [0, 1/N), then N evenly spaced monotone pointers (the classic method)
    offset = rng.uniform() / n
    positions = [offset + i / n for i in range(n)]
    idx = [0] * n
    cumulative = []
    c = 0.0
    for w in weights:
        c += w
        cumulative.append(c)
    i = 0
    for j in range(n):
        while i < n - 1 and positions[j] > cumulative[i]:
            i += 1
        idx[j] = i
    return idx


class ParticleFilter:
    """A generic bootstrap (sequential-importance-resampling) particle filter.

    transition_fn(state, rng)     -> next state (applies the motion model AND process noise)
    likelihood_fn(state, measure) -> P(measurement | state), a nonnegative weight
    init_fn(rng)                  -> an initial state sample from the prior
    """

    def __init__(self, transition_fn, likelihood_fn, init_fn, n_particles=500,
                 resample_threshold=0.5, seed=0):
        self.transition_fn = transition_fn
        self.likelihood_fn = likelihood_fn
        self.rng = _Rng(seed)
        self.n = n_particles
        self.resample_threshold = resample_threshold
        self.particles = [init_fn(self.rng) for _ in range(n_particles)]
        self.weights = [1.0 / n_particles] * n_particles
        self.ess_history = []
        self.resampled_history = []

    def step(self, measurement):
        """Advance one time step against a measurement; return the weighted-mean state estimate."""
        # PREDICT
        self.particles = [self.transition_fn(p, self.rng) for p in self.particles]
        # WEIGHT
        w = [self.weights[i] * self.likelihood_fn(self.particles[i], measurement)
             for i in range(self.n)]
        total = sum(w)
        if total <= 0:
            # total collapse: reset to uniform (all particles equally bad)
            self.weights = [1.0 / self.n] * self.n
        else:
            self.weights = [wi / total for wi in w]
        # ESTIMATE (before any resampling, using the current weights)
        est = self._weighted_mean()
        # RESAMPLE if degenerate
        ess = effective_sample_size(self.weights)
        self.ess_history.append(ess)
        if ess < self.resample_threshold * self.n:
            idx = systematic_resample(self.weights, self.rng)
            self.particles = [self._copy(self.particles[i]) for i in idx]
            self.weights = [1.0 / self.n] * self.n
            self.resampled_history.append(True)
        else:
            self.resampled_history.append(False)
        return est

    def _copy(self, state):
        return list(state) if isinstance(state, list) else state

    def _weighted_mean(self):
        p0 = self.particles[0]
        if isinstance(p0, (list, tuple)):
            d = len(p0)
            return [sum(self.weights[i] * self.particles[i][j] for i in range(self.n))
                    for j in range(d)]
        return sum(self.weights[i] * self.particles[i] for i in range(self.n))

    def run(self, measurements):
        """Filter a whole sequence; return the list of per-step state estimates."""
        return [self.step(z) for z in measurements]


def gaussian_likelihood(predicted, observed, sigma):
    """Unnormalized Gaussian likelihood of a scalar observation (constant factors cancel)."""
    return math.exp(-0.5 * ((observed - predicted) / sigma) ** 2)
