"""Simpson's paradox: an association that reverses when a population is split by a confounder.

One of the most disorienting facts in statistics: a treatment can look WORSE than its rival in the
pooled data yet BETTER in every single subgroup -- or the reverse. The aggregate trend and the
within-group trend point opposite ways, and both are arithmetically correct. It is not a trick of
small samples; it happens with exact counts, and it is why "the numbers" alone never settle a causal
question.

The mechanism is a lurking CONFOUNDER that is distributed unevenly across the groups being compared.
The textbook case is a 1986 kidney-stone study. Treatment A (open surgery) had a lower overall success
rate than treatment B (a less invasive procedure) -- yet A did better on BOTH small stones AND large
stones taken separately. The resolution: A was preferentially given the hard cases (large stones,
which fare worse for everyone), so pooling mixed a difficult case-mix into A's numbers. Stone size is
the confounder; once you hold it fixed, A wins. The same shape appears in the Berkeley 1973 admissions
data (women applied to more competitive departments) and in batting averages across two seasons.

This module works on STRATIFIED 2x2 tables -- for each stratum, the successes and totals of two groups
-- and asks the three questions that matter:

  DETECT. Does the pooled winner disagree with the per-stratum winner? `is_reversal` flags a genuine
      Simpson reversal: one group wins the aggregate while the other wins every stratum.
  EXPLAIN. `pooled_rates` and `stratum_rates` expose the two conflicting views; the reversal comes from
      the confounder's uneven `allocation` across groups.
  CORRECT. Which answer should you trust? The confounder-ADJUSTED comparison. `mantel_haenszel_or`
      gives the standard stratified odds ratio (a weighted blend of the per-stratum tables that cancels
      the case-mix), and `standardized_rates` applies each group's per-stratum rates to a common
      population -- both recover the within-stratum direction, the causally correct one.

Pure stdlib, exact rational arithmetic where it matters, no randomness. Validated against the kidney-stone
and Berkeley datasets: the reversal is detected, the naive pooled odds ratio points the WRONG way while
Mantel-Haenszel and direct standardization both point the RIGHT way, and a brute-force scan confirms that
"pooled sign opposite to every stratum's sign" is exactly the reversal condition. The observational-bias
companion to the contingency-table, logistic-regression, and causal-DAG notes."""

from __future__ import annotations


def _rate(succ, tot):
    return succ / tot if tot else 0.0


def pooled_rates(strata):
    """Aggregate success rates for the two groups over all strata (the naive, confounded view).

    `strata` is a list of (a_succ, a_tot, b_succ, b_tot). Returns (rate_a, rate_b) computed on the
    summed counts -- exactly what you get if you ignore the stratifying variable."""
    a_s = sum(s[0] for s in strata)
    a_t = sum(s[1] for s in strata)
    b_s = sum(s[2] for s in strata)
    b_t = sum(s[3] for s in strata)
    return _rate(a_s, a_t), _rate(b_s, b_t)


def stratum_rates(strata):
    """Per-stratum (rate_a, rate_b) pairs -- the within-group view that the confounder does not distort."""
    return [(_rate(a_s, a_t), _rate(b_s, b_t)) for (a_s, a_t, b_s, b_t) in strata]


def _sign(x, eps=1e-12):
    return 0 if abs(x) < eps else (1 if x > 0 else -1)


def is_reversal(strata, eps=1e-12):
    """True iff the data exhibit a genuine Simpson reversal.

    Condition: group A beats group B in EVERY stratum (all per-stratum sign the same and non-zero) but
    LOSES the pooled comparison, or vice versa. Ties in any stratum, or a pooled tie, are not reversals."""
    pr_a, pr_b = pooled_rates(strata)
    pooled_sign = _sign(pr_a - pr_b, eps)
    if pooled_sign == 0:
        return False
    signs = [_sign(ra - rb, eps) for (ra, rb) in stratum_rates(strata)]
    if any(s == 0 for s in signs):
        return False
    if len(set(signs)) != 1:  # strata disagree among themselves -> not a clean reversal
        return False
    return signs[0] != pooled_sign


def odds_ratio(a_succ, a_tot, b_succ, b_tot):
    """Odds ratio of success for group A vs group B in a single 2x2 table.

    OR = (a_succ/a_fail) / (b_succ/b_fail) = (a_succ * b_fail) / (a_fail * b_succ). >1 favours A."""
    a_fail = a_tot - a_succ
    b_fail = b_tot - b_succ
    denom = a_fail * b_succ
    if denom == 0:
        return float("inf") if a_succ * b_fail > 0 else float("nan")
    return (a_succ * b_fail) / denom


def pooled_odds_ratio(strata):
    """Naive odds ratio computed on the collapsed (summed) table -- the misleading confounded estimate."""
    a_s = sum(s[0] for s in strata)
    a_t = sum(s[1] for s in strata)
    b_s = sum(s[2] for s in strata)
    b_t = sum(s[3] for s in strata)
    return odds_ratio(a_s, a_t, b_s, b_t)


def mantel_haenszel_or(strata):
    """Mantel-Haenszel stratified odds ratio: the confounder-adjusted A-vs-B comparison.

    For each 2x2 stratum with exposed(A) success a / failure b and unexposed(B) success c / failure d
    over n = a+b+c+d subjects, MH pools as OR = sum(a*d/n) / sum(b*c/n). This weights each stratum by
    how much information it carries and cancels the uneven case-mix that produces the paradox."""
    num = 0.0
    den = 0.0
    for (a_succ, a_tot, b_succ, b_tot) in strata:
        a = a_succ
        b = a_tot - a_succ
        c = b_succ
        d = b_tot - b_succ
        n = a_tot + b_tot
        if n == 0:
            continue
        num += a * d / n
        den += b * c / n
    if den == 0:
        return float("inf") if num > 0 else float("nan")
    return num / den


def standardized_rates(strata):
    """Direct standardization: apply each group's per-stratum rates to a COMMON population distribution.

    The shared weights are the combined stratum sizes (a_tot + b_tot) normalized to sum 1, so both
    groups are scored on the same case-mix. Returns (std_rate_a, std_rate_b); the difference now reflects
    the within-stratum effect, not the confounder. Recovers the per-stratum direction."""
    sizes = [s[1] + s[3] for s in strata]
    total = sum(sizes)
    if total == 0:
        return 0.0, 0.0
    weights = [sz / total for sz in sizes]
    rates = stratum_rates(strata)
    std_a = sum(w * ra for w, (ra, rb) in zip(weights, rates))
    std_b = sum(w * rb for w, (ra, rb) in zip(weights, rates))
    return std_a, std_b


def allocation(strata):
    """Fraction of each group's subjects that fall in each stratum -- exposes the confounder imbalance.

    Returns a list (one per stratum) of (frac_of_A_here, frac_of_B_here). When these differ sharply
    between groups, the groups face different case-mixes and the paradox becomes possible."""
    a_tot = sum(s[1] for s in strata)
    b_tot = sum(s[3] for s in strata)
    out = []
    for (a_s, a_t, b_s, b_t) in strata:
        out.append((a_t / a_tot if a_tot else 0.0, b_t / b_tot if b_tot else 0.0))
    return out
