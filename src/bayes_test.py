"""Bayes and the base-rate fallacy: why a positive test can still mean healthy.

A disease affects 1 in 1000 people. A test is 99% sensitive (it catches 99% of the sick) and
99% specific (it clears 99% of the healthy). You test positive. What is the chance you are
actually sick? Almost everyone -- including most doctors, in the famous studies -- guesses 99%.
The right answer is about 9%.

The reason is the base rate. Bayes' theorem combines the prior (how common the disease is) with
the test's likelihoods:

    P(sick | +) = P(+ | sick) P(sick) / P(+),
    P(+) = P(+ | sick) P(sick) + P(+ | healthy) P(healthy).

With a prior of 0.001, out of 100,000 people only 100 are sick (99 test positive) but 99,900
are healthy and 1% of them -- 999 people -- test positive anyway. So among 1098 positives only
99 are truly sick: 9%. The rare disease makes false positives swamp the true ones no matter how
good the test sounds. This is the base-rate fallacy, and it governs medical screening, spam
filters, airport-security profiling, and any rare-event detector.

This module computes the posterior (positive and negative predictive value) from a prior,
sensitivity, and specificity via Bayes' theorem, the likelihood ratios, the effect of
retesting, the prevalence at which a positive becomes more-likely-than-not, and a seeded
Monte-Carlo cohort that confirms the counts. Pure stdlib; the conditional-probability companion
to the Monty Hall and birthday notes.
"""

from __future__ import annotations


def posterior_positive(prior: float, sensitivity: float, specificity: float) -> float:
    """P(disease | positive test), the positive predictive value, by Bayes' theorem.

        = sens * prior / (sens * prior + (1 - spec) * (1 - prior))."""
    _validate(prior, sensitivity, specificity)
    tp = sensitivity * prior
    fp = (1.0 - specificity) * (1.0 - prior)
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def posterior_negative(prior: float, sensitivity: float, specificity: float) -> float:
    """P(disease | negative test) -- the residual chance of disease after a clear result.

        = (1 - sens) * prior / ((1 - sens) * prior + spec * (1 - prior))."""
    _validate(prior, sensitivity, specificity)
    fn = (1.0 - sensitivity) * prior
    tn = specificity * (1.0 - prior)
    denom = fn + tn
    return fn / denom if denom > 0 else 0.0


def negative_predictive_value(prior: float, sensitivity: float, specificity: float) -> float:
    """P(healthy | negative test) = 1 - posterior_negative."""
    return 1.0 - posterior_negative(prior, sensitivity, specificity)


def positive_likelihood_ratio(sensitivity: float, specificity: float) -> float:
    """LR+ = sensitivity / (1 - specificity): the factor by which a positive multiplies the
    prior odds of disease."""
    denom = 1.0 - specificity
    return float("inf") if denom == 0 else sensitivity / denom


def negative_likelihood_ratio(sensitivity: float, specificity: float) -> float:
    """LR- = (1 - sensitivity) / specificity: the factor by which a negative multiplies the
    prior odds of disease."""
    if specificity == 0:
        return float("inf")
    return (1.0 - sensitivity) / specificity


def posterior_from_odds(prior: float, likelihood_ratio: float) -> float:
    """Apply a likelihood ratio to a prior in odds form and convert back to a probability:
    posterior_odds = prior_odds * LR. Handy for chaining independent tests."""
    if not 0.0 < prior < 1.0:
        if prior in (0.0, 1.0):
            return prior
        raise ValueError("prior must be in [0, 1]")
    prior_odds = prior / (1.0 - prior)
    post_odds = prior_odds * likelihood_ratio
    return post_odds / (1.0 + post_odds)


def posterior_after_retests(prior: float, sensitivity: float, specificity: float,
                            n_positive: int) -> float:
    """Posterior after `n_positive` independent positive tests, chaining the LR+ in odds space.
    Two positives on a good test finally push a rare-disease posterior above 1/2."""
    _validate(prior, sensitivity, specificity)
    lr = positive_likelihood_ratio(sensitivity, specificity)
    odds = prior / (1.0 - prior)
    odds *= lr ** n_positive
    return odds / (1.0 + odds)


def prevalence_for_even_odds(sensitivity: float, specificity: float) -> float:
    """The disease prevalence at which a single positive test is exactly 50-50 (posterior =
    1/2). Below this the base-rate fallacy dominates; above it a positive is more likely true.

        prior* = (1 - spec) / (sens + 1 - spec)."""
    fp = 1.0 - specificity
    return fp / (sensitivity + fp)


def _validate(prior, sensitivity, specificity):
    for name, v in (("prior", prior), ("sensitivity", sensitivity), ("specificity", specificity)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")


class _Rng:
    """Seeded LCG; high bits (an LCG's low bits are not random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def simulate(prior: float, sensitivity: float, specificity: float, n: int = 200000,
             seed: int = 1) -> float:
    """Monte-Carlo cohort of n people: draw disease status from the prior, a test result from
    the sensitivity/specificity, and return the empirical P(disease | positive)."""
    rng = _Rng(seed)
    tp = 0
    fp = 0
    for _ in range(n):
        sick = rng.random() < prior
        if sick:
            if rng.random() < sensitivity:  # true positive
                tp += 1
        else:
            if rng.random() >= specificity:  # false positive (test says +)
                fp += 1
    return tp / (tp + fp) if (tp + fp) else 0.0
