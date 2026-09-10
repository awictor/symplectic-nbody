"""Radioactive decay: exponential decay, dating, and decay chains.

An unstable nucleus decays at random with a constant probability per unit time, the
decay constant lambda. A population thus falls exponentially,

    N(t) = N0 exp(-lambda t),      lambda = ln 2 / t_half,

with the half-life t_half the time for half the sample to decay. The activity (decays
per second) is A = lambda N. This is the clock behind radiometric dating: a sample's
remaining fraction of a radioisotope gives its age,

    t = t_half * log2(N0 / N),

so carbon-14 (t_half = 5730 yr) dates organic material back ~50,000 years, and
uranium-lead dates rocks over billions.

When a parent decays into a radioactive daughter, the daughter's abundance follows the
Bateman equation, rising then falling. If the parent is much longer-lived than the
daughter the system reaches SECULAR EQUILIBRIUM, where the daughter's activity equals
the parent's (lambda_D N_D = lambda_P N_P) -- the principle behind radioisotope
generators (e.g. Mo-99/Tc-99m) and the radon that seeps from long-lived uranium.

This module gives the decay constant, the surviving amount and activity, the
radiometric age, the two-step Bateman daughter abundance, and the secular-equilibrium
activity, and reproduces carbon-14 dating and secular equilibrium. SI-friendly (seconds
internally, years helper). Pure stdlib; the nuclear-timescale companion to the Gamow
and pulsar modules.
"""

from __future__ import annotations

import math

YEAR = 3.15576e7
LN2 = math.log(2.0)


def decay_constant(t_half: float) -> float:
    """Decay constant lambda = ln2 / t_half (per unit time; same time unit as t_half)."""
    return LN2 / t_half


def remaining(N0: float, t: float, t_half: float) -> float:
    """Amount remaining after time t: N = N0 exp(-lambda t) = N0 2^(-t/t_half)."""
    return N0 * 2.0 ** (-t / t_half)


def activity(N: float, t_half: float) -> float:
    """Activity A = lambda N (decays per unit time)."""
    return decay_constant(t_half) * N


def age_from_fraction(fraction_remaining: float, t_half: float) -> float:
    """Radiometric age from the surviving fraction N/N0: t = t_half log2(1/fraction)."""
    return t_half * math.log2(1.0 / fraction_remaining)


def n_half_lives(fraction_remaining: float) -> float:
    """Number of half-lives elapsed for a given surviving fraction: log2(1/fraction)."""
    return math.log2(1.0 / fraction_remaining)


def bateman_daughter(N0_parent: float, t: float, t_half_p: float,
                     t_half_d: float) -> float:
    """Daughter abundance in a parent->daughter->stable chain (Bateman, daughter
    starting at zero):
    N_D(t) = N0_P lambda_P / (lambda_D - lambda_P) (exp(-lambda_P t) - exp(-lambda_D t))."""
    lp = decay_constant(t_half_p)
    ld = decay_constant(t_half_d)
    if abs(ld - lp) < 1e-300:
        # equal-rate limit: N_D = N0 lambda t exp(-lambda t)
        return N0_parent * lp * t * math.exp(-lp * t)
    return (N0_parent * lp / (ld - lp)
            * (math.exp(-lp * t) - math.exp(-ld * t)))


def secular_equilibrium_activity(N_parent: float, t_half_p: float) -> float:
    """In secular equilibrium (long-lived parent) the daughter activity equals the
    parent activity, A = lambda_P N_P -- independent of the daughter's half-life."""
    return activity(N_parent, t_half_p)
