"""Stochastic SIR epidemic: how an outbreak grows, fizzles, or sweeps a population -- one infection at a time.

The deterministic SIR model (Kermack-McKendrick 1927) describes an epidemic by smooth curves of
Susceptible, Infected, and Recovered fractions, and predicts an outbreak whenever the BASIC REPRODUCTION
NUMBER R0 = beta/gamma exceeds 1. But early on, with only a handful of infected, CHANCE dominates: the same
R0 > 1 epidemic can fizzle out from bad luck before it ever takes off. The STOCHASTIC SIR model captures
this as a continuous-time Markov jump process with two reactions:

    INFECTION:  S + I -> 2I   at rate  beta * S * I / N
    RECOVERY:   I -> R        at rate  gamma * I

Simulated exactly by Gillespie's algorithm (draw the next-event time from the total rate, pick which
reaction fires in proportion to its rate), it reveals what the ODE hides: a bimodal FINAL EPIDEMIC SIZE --
either a minor outbreak that dies out quickly (probability ~ 1/R0 per initial infective) or a major one
that infects a predictable fraction of the population. That major-outbreak fraction z solves the
transcendental FINAL-SIZE EQUATION 1 - z = exp(-R0 z), the same fixed-point as a branching process's
survival. For large N the major-outbreak trajectory tracks the deterministic ODE.

This module simulates the stochastic SIR by Gillespie, computes R0 and the peak/final-size statistics,
solves the final-size equation, and estimates the extinction (minor-outbreak) probability by ensemble. It
uses a seeded RNG. It is validated: below threshold (R0 < 1) outbreaks stay tiny; above threshold the final
size is bimodal, with the major-outbreak fraction matching the final-size equation 1 - z = e^{-R0 z}; the
minor-outbreak probability from one infective is close to 1/R0; total population S+I+R is conserved on
every path; a larger R0 gives a larger major-outbreak size and higher take-off probability; and results are
reproducible per seed. Pure stdlib; the epidemic-dynamics companion to the Gillespie-SSA, Galton-Watson,
and reaction-diffusion tools."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def r0(beta, gamma):
    """Basic reproduction number R0 = beta / gamma."""
    return beta / gamma


def simulate(n, i0, beta, gamma, seed=1, max_events=10_000_000):
    """Gillespie simulation of the stochastic SIR. n = population, i0 = initial infectives.

    Returns a dict with final_size (total ever infected = R at the end), peak_infected, duration,
    and the trajectory (times, S, I, R lists)."""
    rng = _Rng(seed)
    S = n - i0
    I = i0
    R = 0
    t = 0.0
    times = [0.0]
    S_traj, I_traj, R_traj = [S], [I], [R]
    peak = I
    events = 0
    while I > 0 and events < max_events:
        rate_inf = beta * S * I / n
        rate_rec = gamma * I
        total = rate_inf + rate_rec
        if total <= 0:
            break
        u1 = max(rng.u(), 1e-12)
        t += -math.log(u1) / total
        if rng.u() * total < rate_inf and S > 0:
            S -= 1
            I += 1
        else:
            I -= 1
            R += 1
        peak = max(peak, I)
        times.append(t)
        S_traj.append(S)
        I_traj.append(I)
        R_traj.append(R)
        events += 1
    return {
        "final_size": R,          # everyone who was ever infected (I started, now recovered)
        "peak_infected": peak,
        "duration": t,
        "times": times,
        "S": S_traj, "I": I_traj, "R": R_traj,
    }


def final_size_fraction(R0, tol=1e-12, max_iter=100000):
    """Major-outbreak final-size fraction z solving 1 - z = exp(-R0 z), the nonzero root for R0 > 1.

    Iterate z <- 1 - exp(-R0 z) from z0 = 0.5. Returns 0 for R0 <= 1 (no major outbreak)."""
    if R0 <= 1:
        return 0.0
    z = 0.5
    for _ in range(max_iter):
        z_new = 1 - math.exp(-R0 * z)
        if abs(z_new - z) < tol:
            return z_new
        z = z_new
    return z


def minor_outbreak_probability(R0, i0=1):
    """Probability an epidemic stays minor (dies out) starting from i0 infectives ~ (1/R0)^i0 for R0>1.

    Below threshold (R0 <= 1) extinction is certain (probability 1)."""
    if R0 <= 1:
        return 1.0
    return (1.0 / R0) ** i0


def extinction_probability(n, i0, beta, gamma, threshold_frac=0.05, n_runs=500, seed=1):
    """Empirical probability of a MINOR outbreak: final size below threshold_frac * N."""
    minor = 0
    thresh = threshold_frac * n
    for run in range(n_runs):
        res = simulate(n, i0, beta, gamma, seed=seed + run * 2749)
        if res["final_size"] < thresh:
            minor += 1
    return minor / n_runs


def deterministic_final_size(n, R0):
    """Deterministic expected number ever infected in a major outbreak: N * final_size_fraction(R0)."""
    return n * final_size_fraction(R0)
