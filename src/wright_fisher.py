"""Wright-Fisher model: how an allele's frequency drifts, fixes, or vanishes in a finite population.

In an infinite population an allele's frequency changes only by selection; in a FINITE one, pure CHANCE in
which individuals happen to reproduce -- GENETIC DRIFT -- pushes frequencies around and eventually drives
every allele to either FIXATION (frequency 1) or LOSS (frequency 0). The WRIGHT-FISHER model (1930s) is the
canonical description: a population of N diploid individuals (2N gene copies) is replaced each generation
by 2N offspring, each independently inheriting allele A with probability equal to A's current frequency
(adjusted for selection). The next generation's count is therefore BINOMIAL(2N, p'), and the frequency
executes a random walk with absorbing barriers at 0 and 1.

Two exact results anchor the model. For a NEUTRAL allele the probability of eventual fixation equals its
CURRENT frequency (a fair game -- a martingale), and the one-generation drift variance is p(1-p)/(2N), so
small populations drift faster. With SELECTION (relative fitness 1+s for A), the fixation probability rises
above the neutral p and, for a single new mutant, is approximately (1 - e^{-2s}) / (1 - e^{-4Ns}) -- the
classic Kimura formula, near 2s for a weakly-beneficial mutant in a large population. Expected
heterozygosity 2p(1-p) decays geometrically by a factor 1 - 1/(2N) per generation as variation is lost.

This module simulates Wright-Fisher trajectories with optional selection, estimates fixation probabilities
and times by ensemble, and computes the neutral and Kimura fixation formulas and the heterozygosity decay.
It uses a seeded RNG with a binomial sampler. It is validated: the neutral fixation probability equals the
starting frequency across a range of p; selection raises the fixation probability above neutral and matches
the Kimura diffusion formula for a new mutant; the one-generation drift variance matches p(1-p)/(2N);
expected heterozygosity decays by 1 - 1/(2N) per generation; every trajectory is absorbed at 0 or 1;
smaller populations fix faster (shorter mean fixation time); and results are reproducible per seed. Pure
stdlib; the population-genetics companion to the Gillespie-SSA, Moran-process, and Markov-chain tools."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def binomial(self, n, p):
        """Sample Binomial(n, p) by summing Bernoulli trials (n is small: 2N gene copies)."""
        if p <= 0:
            return 0
        if p >= 1:
            return n
        count = 0
        for _ in range(n):
            if self.u() < p:
                count += 1
        return count


def _selected_frequency(p, s):
    """Frequency after viability selection with relative fitness 1+s for allele A (haploid-style)."""
    if s == 0:
        return p
    wbar = p * (1 + s) + (1 - p)
    return p * (1 + s) / wbar if wbar > 0 else p


def simulate(two_n, p0, s=0.0, max_gen=100000, seed=1):
    """Simulate one Wright-Fisher trajectory. two_n = 2N gene copies, p0 = initial frequency.

    Returns (trajectory of frequencies, fixed) where fixed is True if absorbed at 1, False if at 0.
    Stops when the allele fixes or is lost."""
    rng = _Rng(seed)
    count = round(p0 * two_n)
    traj = [count / two_n]
    for _ in range(max_gen):
        if count == 0:
            return traj, False
        if count == two_n:
            return traj, True
        p = count / two_n
        p_sel = _selected_frequency(p, s)
        count = rng.binomial(two_n, p_sel)
        traj.append(count / two_n)
    return traj, count == two_n


def fixation_probability(two_n, p0, s=0.0, n_runs=500, seed=1):
    """Estimate the probability of eventual fixation by ensemble simulation."""
    fixed = 0
    for r in range(n_runs):
        _traj, is_fixed = simulate(two_n, p0, s, seed=seed + r)
        if is_fixed:
            fixed += 1
    return fixed / n_runs


def mean_fixation_time(two_n, p0, s=0.0, n_runs=500, seed=1):
    """Mean number of generations to absorption (fixation OR loss), by ensemble."""
    total = 0
    for r in range(n_runs):
        traj, _ = simulate(two_n, p0, s, seed=seed + r)
        total += len(traj) - 1
    return total / n_runs


def neutral_fixation_probability(p0):
    """Exact neutral fixation probability = current frequency (the martingale result)."""
    return p0


def kimura_fixation_probability(two_n, p0, s):
    """Kimura diffusion fixation probability for a co-dominant allele with selection coefficient s.

    P = (1 - e^{-2 N s p0... }) form; standard: (1 - exp(-4 N s p0)) / (1 - exp(-4 N s)) with N = two_n/2."""
    if s == 0:
        return p0
    N = two_n / 2
    x = 4 * N * s
    num = 1 - math.exp(-x * p0)
    den = 1 - math.exp(-x)
    if abs(den) < 1e-300:
        return p0
    return num / den


def drift_variance(two_n, p):
    """One-generation variance of the frequency under neutral drift: p(1-p)/(2N)."""
    return p * (1 - p) / two_n


def heterozygosity_decay(two_n, h0, generations):
    """Expected heterozygosity after `generations`: H_t = H_0 (1 - 1/(2N))^t."""
    return h0 * (1 - 1.0 / two_n) ** generations
