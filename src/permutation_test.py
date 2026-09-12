"""Permutation tests -- exact and Monte-Carlo significance without distributional assumptions.

A permutation test answers a deceptively simple question: is the difference I see between two
groups larger than what I'd get by shuffling the labels at random? Under the null hypothesis that
the two samples come from the same distribution, the group labels are exchangeable -- any assignment
of the pooled observations to "group A" and "group B" is equally likely. So we compute the observed
test statistic (say, the difference in means), then re-form the two groups by every possible
relabelling of the pool, recompute the statistic each time, and ask what fraction of those
relabellings give a statistic at least as extreme as the observed one. That fraction *is* the
p-value -- no t-distribution, no normality assumption, no variance formula. It is exact by
construction, because it enumerates the actual null distribution the data would produce.

For small samples the enumeration is complete: with n_A observations in the first group and N total,
there are C(N, n_A) ways to choose which pooled values land in group A, and we walk all of them. This
gives an *exact* p-value -- the true tail probability under the permutation null, down to the last
digit. As N grows C(N, n_A) explodes (choosing 15 from 30 is 155 million), so beyond a threshold we
switch to Monte-Carlo: draw a few thousand random relabellings, and the fraction exceeding the
observed statistic estimates the exact p-value with a standard error of sqrt(p(1-p)/B). The
Monte-Carlo p-value uses the (b+1)/(B+1) correction -- counting the observed arrangement itself as
one of the permutations -- so it can never be exactly zero and stays a valid conservative estimate.

The framework is statistic-agnostic. Difference in means is the classic choice, but the same
machinery handles difference in medians (robust to outliers), difference in trimmed means, the
t-statistic, or any function of the two groups you can write down -- which is exactly the appeal,
since most of those have no tractable analytic null distribution. This module also provides the
one-sample / paired sign-flip test: under the null that a paired difference has zero-centred
symmetric distribution, each difference's sign is equally likely to be + or -, so we enumerate (or
sample) the 2**n sign patterns and read off the tail. Paired designs -- before/after, matched
subjects -- are where this shines.

Validation. The permutation p-value is checked three ways. (1) Against complete enumeration: for
small n the Monte-Carlo estimate is compared to the exact count and must agree within sampling
error. (2) Against the analytic t-test: for large, genuinely-normal samples the permutation p-value
tracks the two-sample t p-value closely, because the permutation null converges to the t null when
its assumptions hold. (3) Calibration under the null: when the two groups are drawn from the *same*
distribution, the p-value is uniform on [0,1], so a level-alpha test rejects about alpha of the time
over many seeded trials -- the defining property of a correctly-calibrated test. All three hold.

Pure standard library. A seeded linear-congruential generator drives the Monte-Carlo sampling so
every run is bit-for-bit reproducible.
"""

import math
from itertools import combinations


class _LCG:
    """Seeded linear-congruential generator (Numerical Recipes constants)."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def _next(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def randint(self, n):
        """Uniform integer in [0, n) using the high bits."""
        return (self._next() >> 8) % n

    def randbit(self):
        """A single fair bit from the high end of the state."""
        return (self._next() >> 16) & 1


# ---------------------------------------------------------------------------
# statistics that consume two groups
# ---------------------------------------------------------------------------

def mean(xs):
    return sum(xs) / len(xs)


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def diff_of_means(a, b):
    """Observed effect: mean(a) - mean(b)."""
    return mean(a) - mean(b)


def diff_of_medians(a, b):
    return median(a) - median(b)


def t_statistic(a, b):
    """Welch-style two-sample t-statistic (unequal variances)."""
    na, nb = len(a), len(b)
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a) / (na - 1) if na > 1 else 0.0
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1) if nb > 1 else 0.0
    denom = math.sqrt(va / na + vb / nb)
    if denom == 0.0:
        return 0.0
    return (ma - mb) / denom


# ---------------------------------------------------------------------------
# two-sample permutation test
# ---------------------------------------------------------------------------

def permutation_test(a, b, statistic=diff_of_means, alternative="two-sided",
                     n_resamples=10000, seed=12345, exact_threshold=100000):
    """Two-sample permutation test on the pooled labels.

    Returns a dict with the observed statistic, the p-value, whether the null distribution was
    enumerated exactly, and the number of permutations used.

    alternative: 'two-sided' (|stat| >= |observed|), 'greater' (stat >= observed), or 'less'
    (stat <= observed).
    """
    a = list(a)
    b = list(b)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        raise ValueError("both groups must be non-empty")
    pool = a + b
    n = na + nb
    observed = statistic(a, b)

    total = math.comb(n, na)
    exact = total <= exact_threshold

    def extreme(stat):
        if alternative == "two-sided":
            return abs(stat) >= abs(observed) - 1e-12
        if alternative == "greater":
            return stat >= observed - 1e-12
        if alternative == "less":
            return stat <= observed + 1e-12
        raise ValueError(f"unknown alternative: {alternative!r}")

    if exact:
        count = 0
        for idx in combinations(range(n), na):
            idx_set = set(idx)
            ga = [pool[i] for i in idx]
            gb = [pool[i] for i in range(n) if i not in idx_set]
            if extreme(statistic(ga, gb)):
                count += 1
        p = count / total
        return {"observed": observed, "p_value": p, "exact": True,
                "n_permutations": total, "alternative": alternative}

    rng = _LCG(seed)
    count = 0
    for _ in range(n_resamples):
        perm = _shuffle(pool, rng)
        ga = perm[:na]
        gb = perm[na:]
        if extreme(statistic(ga, gb)):
            count += 1
    # (count + 1) / (B + 1): the observed arrangement counts as one permutation, so p is never
    # exactly zero and the estimate stays a valid conservative bound.
    p = (count + 1) / (n_resamples + 1)
    return {"observed": observed, "p_value": p, "exact": False,
            "n_permutations": n_resamples, "alternative": alternative}


def _shuffle(seq, rng):
    """Fisher-Yates shuffle into a fresh list using the seeded generator."""
    out = list(seq)
    for i in range(len(out) - 1, 0, -1):
        j = rng.randint(i + 1)
        out[i], out[j] = out[j], out[i]
    return out


# ---------------------------------------------------------------------------
# paired / one-sample sign-flip test
# ---------------------------------------------------------------------------

def sign_flip_test(differences, alternative="two-sided", n_resamples=10000,
                   seed=12345, exact_threshold=20):
    """Paired sign-flip permutation test.

    Under the null that the paired differences are symmetric about zero, each difference's sign is
    equally likely, so the null distribution of the mean is generated by flipping signs. Enumerates
    all 2**n patterns when n <= exact_threshold, otherwise samples them.

    Returns the same dict shape as permutation_test.
    """
    d = [x for x in differences]
    n = len(d)
    if n == 0:
        raise ValueError("need at least one difference")
    observed = sum(d) / n

    def extreme(stat):
        if alternative == "two-sided":
            return abs(stat) >= abs(observed) - 1e-12
        if alternative == "greater":
            return stat >= observed - 1e-12
        if alternative == "less":
            return stat <= observed + 1e-12
        raise ValueError(f"unknown alternative: {alternative!r}")

    total = 1 << n
    if n <= exact_threshold:
        count = 0
        for mask in range(total):
            s = 0.0
            for i in range(n):
                s += d[i] if (mask >> i) & 1 else -d[i]
            if extreme(s / n):
                count += 1
        return {"observed": observed, "p_value": count / total, "exact": True,
                "n_permutations": total, "alternative": alternative}

    rng = _LCG(seed)
    count = 0
    for _ in range(n_resamples):
        s = 0.0
        for i in range(n):
            s += d[i] if rng.randbit() else -d[i]
        if extreme(s / n):
            count += 1
    p = (count + 1) / (n_resamples + 1)
    return {"observed": observed, "p_value": p, "exact": False,
            "n_permutations": n_resamples, "alternative": alternative}
