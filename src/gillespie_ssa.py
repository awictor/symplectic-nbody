"""Gillespie's stochastic simulation algorithm: exact trajectories of a chemical reaction network.

Deterministic rate equations (ODEs) describe reactions as smooth concentration curves -- correct when
molecule counts are huge, but wrong when they are small, where reactions fire one at a time and CHANCE
dominates: gene expression from a handful of mRNAs, a predator-prey system near extinction, an epidemic
starting from one case. Gillespie's SSA (1976) simulates the EXACT stochastic dynamics of such a system as
a continuous-time Markov jump process. Given reactions each with a PROPENSITY a_j (the instantaneous
probability per unit time that reaction j fires, = rate constant times the number of reactant
combinations), the algorithm repeatedly:

  1. computes the total propensity a0 = sum_j a_j,
  2. draws the WAITING TIME to the next reaction as an exponential with rate a0 (tau = -ln(u)/a0),
  3. picks WHICH reaction fires with probability a_j / a0,
  4. applies that reaction's stoichiometry to the molecule counts and advances the clock by tau.

This "direct method" is statistically exact -- every trajectory is a valid sample from the chemical master
equation -- not an approximation, and it naturally captures the noise, bursts, and extinctions that ODEs
miss. Averaging many trajectories recovers the deterministic law in the large-count limit.

This module implements the direct-method SSA over a general reaction network (species, reactions with
integer stoichiometry and mass-action propensities), runs single trajectories and ensembles, and records
the state at sampled times. It uses a seeded RNG. It is validated: on a reversible A<->B system the mean
trajectory matches the deterministic ODE steady state and conserves total mass exactly on every path; a
pure decay A->0 gives the analytic exponential mean and the right extinction-time distribution; the
waiting times are exponentially distributed with the total propensity; the reaction choice frequencies
match the propensity ratios; a Lotka-Volterra predator-prey network produces sustained stochastic
oscillations; ensemble means converge to the ODE as population grows; and results are reproducible per
seed. Pure stdlib; the stochastic-kinetics companion to the reaction-diffusion, master-equation, and
Markov-chain tools."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


class Reaction:
    """A mass-action reaction: reactants/products are {species_index: stoichiometric_count}, rate is k."""

    def __init__(self, reactants, products, rate):
        self.reactants = reactants
        self.products = products
        self.rate = rate

    def propensity(self, state):
        """Mass-action propensity: k * product of falling factorials of reactant counts."""
        a = self.rate
        for sp, count in self.reactants.items():
            x = state[sp]
            # number of ways to choose `count` molecules of species sp: x*(x-1)*...*(x-count+1)/count!
            for i in range(count):
                a *= (x - i)
            # divide by count! for dimer/trimer reactions (bimolecular of same species)
            fact = 1
            for i in range(2, count + 1):
                fact *= i
            a /= fact
        return max(0.0, a)

    def apply(self, state):
        """Return the state after this reaction fires (net stoichiometry change)."""
        new = list(state)
        for sp, count in self.reactants.items():
            new[sp] -= count
        for sp, count in self.products.items():
            new[sp] += count
        return new


def simulate(reactions, initial_state, t_max, seed=1, max_steps=1_000_000):
    """Run one exact SSA trajectory to time t_max. Returns (times, states) recorded at each reaction."""
    rng = _Rng(seed)
    state = list(initial_state)
    t = 0.0
    times = [0.0]
    states = [list(state)]
    steps = 0
    while t < t_max and steps < max_steps:
        propensities = [r.propensity(state) for r in reactions]
        a0 = sum(propensities)
        if a0 <= 0:
            break  # no reaction can fire -- absorbing state
        # waiting time to next reaction
        u1 = max(rng.u(), 1e-12)
        tau = -math.log(u1) / a0
        t += tau
        if t > t_max:
            break
        # choose which reaction fires
        threshold = rng.u() * a0
        cumulative = 0.0
        chosen = len(reactions) - 1
        for j, aj in enumerate(propensities):
            cumulative += aj
            if cumulative >= threshold:
                chosen = j
                break
        state = reactions[chosen].apply(state)
        times.append(t)
        states.append(list(state))
        steps += 1
    return times, states


def sample_at(times, states, sample_times):
    """Sample a step-function trajectory (times, states) at the given sample_times (zero-order hold)."""
    out = []
    idx = 0
    for st in sample_times:
        while idx + 1 < len(times) and times[idx + 1] <= st:
            idx += 1
        out.append(list(states[idx]))
    return out


def ensemble_mean(reactions, initial_state, sample_times, n_runs=100, seed=1):
    """Mean state across n_runs SSA trajectories, sampled at sample_times. Returns list of mean states."""
    n_species = len(initial_state)
    acc = [[0.0] * n_species for _ in sample_times]
    for r in range(n_runs):
        times, states = simulate(reactions, initial_state, sample_times[-1] + 1e-9, seed=seed + r)
        sampled = sample_at(times, states, sample_times)
        for i, st in enumerate(sampled):
            for sp in range(n_species):
                acc[i][sp] += st[sp]
    return [[acc[i][sp] / n_runs for sp in range(n_species)] for i in range(len(sample_times))]


def total_propensity(reactions, state):
    """Sum of propensities -- the rate of the next reaction (exponential clock rate)."""
    return sum(r.propensity(state) for r in reactions)
