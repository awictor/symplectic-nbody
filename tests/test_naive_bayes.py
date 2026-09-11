"""Tests for naive_bayes: Gaussian & multinomial, exact posterior, smoothing, text classification."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from naive_bayes import GaussianNB, MultinomialNB, _logsumexp

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- logsumexp -------------------------------------------------------------
check("logsumexp basic", approx(_logsumexp([0.0, 0.0]), math.log(2.0), 1e-12))
check("logsumexp stable for large values",
      approx(_logsumexp([1000.0, 1000.0]), 1000.0 + math.log(2.0), 1e-9))

# --- LCG data --------------------------------------------------------------
state = 3


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- Gaussian NB separates well-separated blobs ----------------------------
X, y = [], []
for c, (cx, cy) in enumerate([(0.0, 0.0), (5.0, 5.0)]):
    for _ in range(40):
        X.append([cx + (rng() - 0.5) * 2, cy + (rng() - 0.5) * 2])
        y.append(c)
gnb = GaussianNB().fit(X, y)
check("Gaussian NB separates blobs", gnb.accuracy(X, y) == 1.0)

# --- probabilities are valid ------------------------------------------------
pp = gnb.predict_proba([[0.0, 0.0], [5.0, 5.0], [2.5, 2.5]])
check("Gaussian proba rows sum to 1", all(approx(sum(p.values()), 1.0, 1e-9) for p in pp))
check("Gaussian proba in [0,1]", all(0.0 <= v <= 1.0 for p in pp for v in p.values()))
check("confident at class 0 centre", pp[0][0] > 0.99)
check("confident at class 1 centre", pp[1][1] > 0.99)

# --- Gaussian posterior matches an exact hand computation ------------------
# class 0 ~ N(0, 1), class 1 ~ N(4, 1), equal priors
Xe = [[-1.0], [1.0]] * 10 + [[3.0], [5.0]] * 10   # gives mean 0 var 1 / mean 4 var 1
ye = [0] * 20 + [1] * 20
ge = GaussianNB(var_smoothing=0.0).fit(Xe, ye)
check("estimated means correct", approx(ge.theta[0][0], 0.0, 1e-9) and approx(ge.theta[1][0], 4.0, 1e-9))
check("estimated variances correct", approx(ge.var[0][0], 1.0, 1e-9) and approx(ge.var[1][0], 1.0, 1e-9))


def nd(x, m):
    return math.exp(-0.5 * (x - m) ** 2) / math.sqrt(2 * math.pi)


p0 = 0.5 * nd(1.0, 0.0)
p1 = 0.5 * nd(1.0, 4.0)
post0 = ge.predict_proba([[1.0]])[0][0]
check("Gaussian posterior matches analytic Bayes", approx(post0, p0 / (p0 + p1), 1e-6))

# --- priors reflect class frequencies --------------------------------------
Ximb = [[0.0]] * 30 + [[10.0]] * 10
yimb = [0] * 30 + [1] * 10
gi = GaussianNB().fit(Ximb, yimb)
check("prior reflects class 0 frequency", approx(math.exp(gi.log_prior[0]), 0.75, 1e-9))
check("prior reflects class 1 frequency", approx(math.exp(gi.log_prior[1]), 0.25, 1e-9))

# --- three-class Gaussian --------------------------------------------------
X3, y3 = [], []
for c, (cx, cy) in enumerate([(0.0, 0.0), (6.0, 0.0), (3.0, 6.0)]):
    for _ in range(25):
        X3.append([cx + (rng() - 0.5) * 1.5, cy + (rng() - 0.5) * 1.5])
        y3.append(c)
g3 = GaussianNB().fit(X3, y3)
check("three-class Gaussian high accuracy", g3.accuracy(X3, y3) >= 0.95)

# --- Multinomial NB: spam vs ham text --------------------------------------
# vocab: [free, money, meeting, report]
Xt = [[3, 2, 0, 0], [2, 3, 0, 1], [4, 1, 0, 0],
      [0, 0, 3, 2], [0, 1, 2, 3], [0, 0, 4, 1]]
yt = ["spam", "spam", "spam", "ham", "ham", "ham"]
mnb = MultinomialNB(alpha=1.0).fit(Xt, yt)
check("multinomial fits training text", mnb.accuracy(Xt, yt) == 1.0)
check("spammy doc -> spam", mnb.predict([[2, 2, 0, 0]])[0] == "spam")
check("meeting doc -> ham", mnb.predict([[0, 0, 2, 2]])[0] == "ham")

# --- Laplace smoothing prevents zero probability ---------------------------
mp = mnb.predict_proba([[0, 0, 0, 0]])[0]
check("smoothing keeps probabilities finite", all(math.isfinite(v) for v in mp.values()))
check("smoothed multinomial proba sums to 1", approx(sum(mp.values()), 1.0, 1e-9))
# a word never seen in a class still gets nonzero probability
check("unseen word has nonzero prob", all(math.exp(lp) > 0 for lp in mnb.log_prob["spam"]))

# --- higher smoothing pulls class word-distributions toward uniform --------
mnb_lo = MultinomialNB(alpha=0.01).fit(Xt, yt)
mnb_hi = MultinomialNB(alpha=100.0).fit(Xt, yt)
spread_lo = max(mnb_lo.log_prob["spam"]) - min(mnb_lo.log_prob["spam"])
spread_hi = max(mnb_hi.log_prob["spam"]) - min(mnb_hi.log_prob["spam"])
check("more smoothing flattens word distribution", spread_hi < spread_lo)

# --- multinomial probabilities normalized ----------------------------------
allp = mnb.predict_proba(Xt)
check("all multinomial rows sum to 1", all(approx(sum(p.values()), 1.0, 1e-9) for p in allp))

# --- determinism -----------------------------------------------------------
check("Gaussian deterministic", gnb.predict(X) == gnb.predict(X))
check("multinomial deterministic", mnb.predict(Xt) == mnb.predict(Xt))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all naive_bayes tests passed")
