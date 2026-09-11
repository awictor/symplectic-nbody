"""Mutual information: measuring how much two variables tell you about each other.

Entropy H(X) measures the uncertainty in one variable. MUTUAL INFORMATION I(X;Y) measures how much
knowing one REDUCES the uncertainty in the other -- the shared information between them. It is the
gold-standard dependency measure because, unlike correlation, it captures ANY relationship, linear
or not: I(X;Y) = 0 exactly when X and Y are independent, and grows with how tightly they are
coupled. It underlies feature selection (keep the features most informative about the label),
decision-tree splits (information gain), image registration, and independent-component analysis.

Everything is built from Shannon entropy H = -sum p log2 p (bits):

    I(X;Y) = H(X) + H(Y) - H(X,Y)        shared information
           = H(X) - H(X|Y)               reduction in X's uncertainty from knowing Y
    H(X|Y) = H(X,Y) - H(Y)               conditional entropy: uncertainty left in X given Y

Two normalizations put I on a 0..1 scale so datasets are comparable, and the KULLBACK-LEIBLER
divergence D(p||q) = sum p log2(p/q) -- the extra bits to code samples from p using a code built
for q -- is the asymmetric "distance" from which mutual information is the divergence of the joint
from the product of marginals. INFORMATION GAIN, the decision-tree split criterion, is exactly the
mutual information between a feature and the label.

This module estimates entropies and mutual information from samples or a joint distribution, with
KL divergence and normalized MI -- verified that independent variables have zero mutual information,
that a deterministic copy has I = H (maximal), that I(X;Y) matches H(X)+H(Y)-H(X,Y) and the
conditional-entropy identity, that KL is nonnegative and zero only for equal distributions, and
against hand-computed values. Pure stdlib; the dependency-measure companion to the Shannon-coding
note."""

from __future__ import annotations

import math


def _log2(x):
    return math.log(x) / math.log(2.0)


def entropy(probs):
    """Shannon entropy H = -sum p log2 p (bits) of a probability list."""
    return -sum(p * _log2(p) for p in probs if p > 0)


def entropy_counts(counts):
    """Entropy from raw counts (normalized internally)."""
    total = sum(counts)
    if total == 0:
        return 0.0
    return entropy([c / total for c in counts])


def _joint_counts(xs, ys):
    """Joint count table {(x,y): n} and the marginals, from paired samples."""
    joint = {}
    cx = {}
    cy = {}
    for x, y in zip(xs, ys):
        joint[(x, y)] = joint.get((x, y), 0) + 1
        cx[x] = cx.get(x, 0) + 1
        cy[y] = cy.get(y, 0) + 1
    return joint, cx, cy


def entropy_samples(xs):
    """Entropy of a sample sequence (plug-in estimator from empirical frequencies)."""
    c = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return entropy_counts(list(c.values()))


def joint_entropy(xs, ys):
    """H(X,Y) from paired samples."""
    joint, _, _ = _joint_counts(xs, ys)
    return entropy_counts(list(joint.values()))


def conditional_entropy(xs, ys):
    """H(X|Y) = H(X,Y) - H(Y): the uncertainty left in X once Y is known."""
    return joint_entropy(xs, ys) - entropy_samples(ys)


def mutual_information(xs, ys):
    """I(X;Y) = H(X) + H(Y) - H(X,Y) in bits, from paired samples (>= 0)."""
    n = len(xs)
    joint, cx, cy = _joint_counts(xs, ys)
    mi = 0.0
    for (x, y), nxy in joint.items():
        pxy = nxy / n
        px = cx[x] / n
        py = cy[y] / n
        mi += pxy * _log2(pxy / (px * py))
    return max(0.0, mi)      # clamp tiny negative round-off


def normalized_mutual_information(xs, ys, method="sqrt"):
    """MI scaled to [0, 1]. method: 'sqrt' -> I/sqrt(H(X)H(Y)), 'min' -> I/min(H(X),H(Y)),
    'sum' -> 2I/(H(X)+H(Y)) (the symmetric-uncertainty coefficient)."""
    hx = entropy_samples(xs)
    hy = entropy_samples(ys)
    mi = mutual_information(xs, ys)
    if hx == 0 or hy == 0:
        return 0.0
    if method == "min":
        return mi / min(hx, hy)
    if method == "sum":
        return 2 * mi / (hx + hy)
    return mi / math.sqrt(hx * hy)


def kl_divergence(p, q):
    """Kullback-Leibler divergence D(p||q) = sum p log2(p/q) in bits (asymmetric, >= 0).

    p and q are probability lists of equal length; q must be positive wherever p is."""
    d = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float("inf")     # p has support where q does not
            d += pi * _log2(pi / qi)
    return d


def information_gain(feature, labels):
    """Information gain of splitting on `feature` to predict `labels` -- exactly I(feature; label),
    the decision-tree split criterion."""
    return mutual_information(labels, feature)


def mutual_information_from_joint(joint):
    """I(X;Y) directly from a joint distribution given as {(x,y): prob} (probabilities sum to 1)."""
    px = {}
    py = {}
    for (x, y), p in joint.items():
        px[x] = px.get(x, 0.0) + p
        py[y] = py.get(y, 0.0) + p
    mi = 0.0
    for (x, y), pxy in joint.items():
        if pxy > 0:
            mi += pxy * _log2(pxy / (px[x] * py[y]))
    return max(0.0, mi)
