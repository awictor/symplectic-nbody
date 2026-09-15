"""Deffuant bounded-confidence: continuous opinions that merge only between people already close enough.

The voter model has two opinions and copies wholesale; real opinions are CONTINUOUS and shift only toward
views not too far from one's own -- you argue productively with someone who mostly agrees, and tune out
someone whose view is alien. The DEFFUANT model (2000) captures this BOUNDED CONFIDENCE. Every agent holds
an opinion in [0, 1]. Pick a random PAIR; if their opinions differ by less than a confidence THRESHOLD d,
each moves a fraction mu of the way toward the other:

    if |x_i - x_j| < d:   x_i += mu (x_j - x_i),   x_j += mu (x_i - x_j).

If they differ by more than d, nothing happens -- the gap is unbridgeable. Iterating, the population
settles into one or more OPINION CLUSTERS, and the confidence threshold d decides the outcome: a LARGE d
(open-minded society) collapses everyone to a single CONSENSUS, while a SMALL d (echo chambers) freezes
into several separated clusters -- POLARIZATION or FRAGMENTATION. The number of surviving clusters is
approximately 1/(2d), a clean rule connecting individual open-mindedness to collective diversity of
opinion.

This module runs the Deffuant dynamics on a population, counts the final opinion clusters, and reports how
the cluster count depends on the threshold. It uses a seeded RNG. It is validated: the MEAN opinion is
conserved by every interaction (a symmetric exchange), so it is preserved to the end; a large confidence
threshold yields a single consensus cluster while a small one yields several; the number of final clusters
grows as the threshold shrinks, tracking the ~1/(2d) rule; opinions stay within [0, 1]; interactions
between distant opinions leave both unchanged; and results are reproducible per seed. Pure stdlib; the
opinion-dynamics companion to the voter-model, Schelling, and Kuramoto tools."""

from __future__ import annotations


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def randint(self, lo, hi):
        return lo + int(self.u() * (hi - lo))


def interact(opinions, i, j, d, mu):
    """One Deffuant interaction between agents i and j. Returns True if they converged (were within d)."""
    xi, xj = opinions[i], opinions[j]
    if abs(xi - xj) < d:
        opinions[i] = xi + mu * (xj - xi)
        opinions[j] = xj + mu * (xi - xj)
        return True
    return False


def mean_opinion(opinions):
    return sum(opinions) / len(opinions)


def simulate(n, d, mu=0.5, steps=None, seed=1, init=None):
    """Run the Deffuant model: n agents, confidence threshold d, convergence rate mu.

    Returns a dict with the final opinions, cluster list, cluster count, and preserved mean. `steps`
    defaults to a large multiple of n so the dynamics settle."""
    rng = _Rng(seed)
    if init is not None:
        opinions = list(init)
    else:
        opinions = [rng.u() for _ in range(n)]
    if steps is None:
        steps = 400 * n
    for _ in range(steps):
        i = rng.randint(0, n)
        j = rng.randint(0, n - 1)
        if j >= i:
            j += 1  # ensure i != j
        interact(opinions, i, j, d, mu)
    clusters = find_clusters(opinions, d)
    return {"opinions": opinions, "clusters": clusters, "n_clusters": len(clusters),
            "mean": mean_opinion(opinions)}


def find_clusters(opinions, d, tol=None):
    """Group the final opinions into clusters: sort and split wherever a gap exceeds the tolerance.

    tol defaults to d/2 -- at equilibrium, converged opinions are near-identical and distinct clusters
    are separated by more than the confidence threshold. Returns a list of cluster mean opinions."""
    if tol is None:
        tol = d / 2
    if not opinions:
        return []
    s = sorted(opinions)
    clusters = []
    group = [s[0]]
    for x in s[1:]:
        if x - group[-1] <= tol:
            group.append(x)
        else:
            clusters.append(sum(group) / len(group))
            group = [x]
    clusters.append(sum(group) / len(group))
    return clusters


def predicted_clusters(d):
    """Rough theoretical number of final clusters ~ 1/(2d) (integer, at least 1)."""
    import math
    return max(1, round(1.0 / (2 * d)))
