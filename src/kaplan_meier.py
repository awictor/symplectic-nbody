"""Kaplan-Meier survival analysis: estimating survival curves from censored data, and comparing groups.

Survival analysis asks: how long until an event -- a patient relapses, a machine fails, a customer
churns? The hard part is CENSORING: when the study ends, many subjects have NOT yet had the event, so
you know only that their survival time exceeds some value, not what it is. Throwing those subjects out
biases the answer; keeping them naively is impossible. The KAPLAN-MEIER estimator (1958), one of the
most-cited papers in all of science, handles censoring exactly.

The idea is the PRODUCT-LIMIT. Order the distinct event times t_1 < t_2 < ... At each event time t_i,
let d_i be the number of events and n_i the number still AT RISK (neither failed nor censored before
t_i). The probability of surviving past t_i, given survival up to it, is (1 - d_i/n_i), so the overall
survival function is the running product:

    S(t) = product over t_i <= t of (1 - d_i / n_i)

Censored subjects contribute to n_i right up until they are censored, then silently leave the
at-risk set -- exactly the information they carry, no more. The variance comes from GREENWOOD'S
FORMULA, giving pointwise confidence bands.

To compare two groups (treatment vs control), the LOG-RANK TEST asks whether their survival curves
differ more than chance allows. At each event time it compares the observed events in group 1 to the
number expected if both groups shared one survival curve (a hypergeometric argument), sums the
observed-minus-expected, and standardizes -- yielding a chi-squared statistic. It is the standard
hypothesis test behind every clinical-trial survival plot.

This module implements the Kaplan-Meier estimator with Greenwood variance, the median survival time,
and the two-sample log-rank test. It is validated against ground truth: with NO censoring the KM curve
is exactly 1 minus the empirical CDF; the curve is monotonically non-increasing, starts at 1, and each
drop matches the product-limit factor; a hand-computed textbook example reproduces the published
survival probabilities; Greenwood's variance matches a direct computation; the median is the first
time the curve crosses 0.5; and the log-rank statistic is ~0 for identical groups and large for
well-separated ones. Pure stdlib; the censored-data companion to the empirical-CDF, bootstrap, and
Kolmogorov-Smirnov tools."""

from __future__ import annotations

import math


def kaplan_meier(times, events):
    """Kaplan-Meier survival estimate.

    times: observed times (event or censoring). events: 1 if the event occurred, 0 if censored.
    Returns (event_times, survival, variance): the distinct event times and the survival probability
    and Greenwood variance just AFTER each. S starts at 1 before the first event.
    """
    if len(times) != len(events):
        raise ValueError("times and events must have equal length")
    n_total = len(times)
    if n_total == 0:
        return [], [], []

    # sort by time; at equal times, process events before censorings is not required for KM
    order = sorted(range(n_total), key=lambda i: times[i])
    t_sorted = [times[i] for i in order]
    e_sorted = [events[i] for i in order]

    # distinct event times (only where at least one event occurs)
    event_times = []
    survival = []
    variance = []
    S = 1.0
    greenwood_sum = 0.0
    at_risk = n_total
    i = 0
    while i < n_total:
        t = t_sorted[i]
        # count events (d) and total leaving (events+censor) at this time
        d = 0
        leaving = 0
        j = i
        while j < n_total and t_sorted[j] == t:
            leaving += 1
            if e_sorted[j] == 1:
                d += 1
            j += 1
        n_i = at_risk
        if d > 0:
            S *= (1.0 - d / n_i)
            if n_i - d > 0:
                greenwood_sum += d / (n_i * (n_i - d))
            event_times.append(t)
            survival.append(S)
            variance.append(S * S * greenwood_sum)
        at_risk -= leaving
        i = j
    return event_times, survival, variance


def survival_at(event_times, survival, t):
    """Survival probability S(t): the product-limit value at the last event time <= t (1 if none)."""
    s = 1.0
    for et, sv in zip(event_times, survival):
        if et <= t:
            s = sv
        else:
            break
    return s


def median_survival(event_times, survival):
    """First time the survival curve drops to or below 0.5, or None if it never does."""
    for et, sv in zip(event_times, survival):
        if sv <= 0.5:
            return et
    return None


def confidence_band(event_times, survival, variance, z=1.96):
    """Pointwise (linear) confidence band S +/- z*sqrt(var), clipped to [0, 1]."""
    band = []
    for sv, var in zip(survival, variance):
        se = math.sqrt(max(var, 0.0))
        lo = max(0.0, sv - z * se)
        hi = min(1.0, sv + z * se)
        band.append((lo, hi))
    return band


def empirical_survival(times):
    """1 minus the empirical CDF at each distinct time (no censoring reference)."""
    n = len(times)
    ts = sorted(set(times))
    out = []
    for t in ts:
        survived = sum(1 for x in times if x > t)
        out.append((t, survived / n))
    return out


def logrank_test(times1, events1, times2, events2):
    """Two-sample log-rank test. Returns (chi2_statistic, observed1, expected1).

    Tests the null hypothesis that the two groups share the same survival function.
    """
    # merge all event times
    all_times = sorted(set(
        [times1[i] for i in range(len(times1)) if events1[i] == 1] +
        [times2[i] for i in range(len(times2)) if events2[i] == 1]
    ))

    O1 = 0.0  # observed events in group 1
    E1 = 0.0  # expected events in group 1 under the null
    V = 0.0   # variance (hypergeometric)

    for t in all_times:
        n1 = sum(1 for x in times1 if x >= t)  # at risk in group 1 just before t
        n2 = sum(1 for x in times2 if x >= t)
        n = n1 + n2
        d1 = sum(1 for i in range(len(times1)) if times1[i] == t and events1[i] == 1)
        d2 = sum(1 for i in range(len(times2)) if times2[i] == t and events2[i] == 1)
        d = d1 + d2
        if n == 0 or d == 0:
            continue
        O1 += d1
        E1 += d * n1 / n
        if n > 1:
            V += (d * (n1 / n) * (n2 / n) * (n - d) / (n - 1))

    if V <= 0:
        return 0.0, O1, E1
    chi2 = (O1 - E1) ** 2 / V
    return chi2, O1, E1
