"""Tests for bayes_test.py -- Bayes' theorem and the base-rate fallacy.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Posteriors are checked against
hand-computed values and a seeded Monte-Carlo cohort.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import bayes_test as b  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, c, tol):
    return abs(a - c) <= tol


PR, SE, SP = 0.001, 0.99, 0.99  # the classic rare-disease example

# --- the famous base-rate result -------------------------------------------
check("positive predictive value is ~9%, not 99%", approx(b.posterior_positive(PR, SE, SP), 0.0902, 1e-3))
check("a positive is still probably a false alarm", b.posterior_positive(PR, SE, SP) < 0.5)
# hand count: 99 true positives, 999 false positives -> 99/1098
check("PPV equals the exact 99/1098", approx(b.posterior_positive(PR, SE, SP), 99 / 1098, 1e-6))
check("negative predictive value is very high", b.negative_predictive_value(PR, SE, SP) > 0.9999)
check("residual disease after a negative is tiny", b.posterior_negative(PR, SE, SP) < 1e-4)

# --- Bayes consistency ------------------------------------------------------
check("posterior_negative + NPV = 1",
      approx(b.posterior_negative(PR, SE, SP) + b.negative_predictive_value(PR, SE, SP), 1.0, 1e-12))
# a perfectly specific test (spec=1) can never give a false positive -> PPV = 1
check("perfect specificity gives PPV 1", approx(b.posterior_positive(0.001, 0.99, 1.0), 1.0, 1e-12))
# a useless coin-flip test (sens = 1 - spec) leaves the prior unchanged
check("an uninformative test leaves the posterior at the prior",
      approx(b.posterior_positive(0.3, 0.6, 0.4), 0.3, 1e-9))
# prior 0 or 1 stays put
check("a zero prior stays zero", b.posterior_positive(0.0, SE, SP) == 0.0)
check("a certain prior stays one", approx(b.posterior_positive(1.0, SE, SP), 1.0, 1e-12))

# --- likelihood ratios & odds form -----------------------------------------
check("LR+ is sens/(1-spec) = 99", approx(b.positive_likelihood_ratio(SE, SP), 99.0, 1e-9))
check("LR- is (1-sens)/spec ~ 0.0101", approx(b.negative_likelihood_ratio(SE, SP), 0.01 / 0.99, 1e-9))
check("the odds form reproduces the direct posterior",
      approx(b.posterior_from_odds(PR, b.positive_likelihood_ratio(SE, SP)),
             b.posterior_positive(PR, SE, SP), 1e-9))
check("LR+ > 1 for a useful test (positive raises odds)", b.positive_likelihood_ratio(SE, SP) > 1)
check("LR- < 1 for a useful test (negative lowers odds)", b.negative_likelihood_ratio(SE, SP) < 1)

# --- retesting --------------------------------------------------------------
check("one positive equals the single-test posterior",
      approx(b.posterior_after_retests(PR, SE, SP, 1), b.posterior_positive(PR, SE, SP), 1e-9))
check("two independent positives push a rare disease above 1/2",
      b.posterior_after_retests(PR, SE, SP, 2) > 0.5)
check("more positives only increase the posterior",
      b.posterior_after_retests(PR, SE, SP, 3) > b.posterior_after_retests(PR, SE, SP, 2))
check("zero retests returns the prior", approx(b.posterior_after_retests(PR, SE, SP, 0), PR, 1e-12))

# --- the even-odds prevalence ----------------------------------------------
p_star = b.prevalence_for_even_odds(SE, SP)
check("even-odds prevalence is (1-spec)/(sens+1-spec) = 0.01", approx(p_star, 0.01, 1e-9))
check("a positive is 50-50 exactly at that prevalence",
      approx(b.posterior_positive(p_star, SE, SP), 0.5, 1e-9))
check("above the threshold a positive is more likely true than not",
      b.posterior_positive(p_star * 2, SE, SP) > 0.5)
check("below the threshold a positive is more likely false",
      b.posterior_positive(p_star / 2, SE, SP) < 0.5)
# rising prevalence raises the PPV monotonically
check("PPV rises with prevalence",
      b.posterior_positive(0.1, SE, SP) > b.posterior_positive(0.01, SE, SP))

# --- Monte-Carlo agreement --------------------------------------------------
check("simulated PPV matches Bayes (rare disease)",
      approx(b.simulate(PR, SE, SP, n=2000000, seed=7), b.posterior_positive(PR, SE, SP), 0.01))
check("simulated PPV matches Bayes (common disease)",
      approx(b.simulate(0.1, 0.9, 0.9, n=500000, seed=3), b.posterior_positive(0.1, 0.9, 0.9), 0.01))

# --- validation -------------------------------------------------------------
try:
    b.posterior_positive(1.5, SE, SP)
    check("rejects prior out of range", False)
except ValueError:
    check("rejects prior out of range", True)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall bayes_test tests passed")
