"""Fisher-Yates: the only correct way to shuffle.

Shuffling a list -- a deck of cards, a playlist, a randomized trial assignment -- sounds trivial,
and almost everyone gets it wrong the first time. The naive "for each position, swap it with a
RANDOM position anywhere in the array" is biased: it can produce n^n equally likely swap
sequences but only n! permutations, and since n^n is not divisible by n! for n > 2 some
permutations come up more often than others. The Fisher-Yates shuffle (1938; the modern
in-place form is due to Durstenfeld, 1964) fixes this by shrinking the range: to place position
i, swap it with a random position in [i, n) -- only the unshuffled tail. Each of the n!
permutations then results from exactly one sequence of choices, so every ordering is equally
likely.

The same downward sweep gives a partial shuffle: stopping after k steps yields a uniform random
k-sample in random order (sampling without replacement), and running it to the end on the
identity gives a uniform random permutation. Restricting each swap to strictly later positions
(the "sattolo" variant) instead produces a uniform random CYCLIC permutation.

This module implements the correct in-place shuffle, the naive biased one (for contrast), a
partial-shuffle k-sample, and Sattolo's cyclic shuffle, all seeded for reproducibility, and it
verifies uniformity by enumerating every permutation over many trials and comparing to the naive
method's measurable bias. Pure stdlib; the permutation companion to the alias-method and
reservoir-sampling notes.
"""

from __future__ import annotations


class _Rng:
    """Seeded LCG; high bits (an LCG's low bits are not random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def randint(self, k: int) -> int:
        """Uniform int in [0, k)."""
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 16) % k

    def randrange(self, lo: int, hi: int) -> int:
        """Uniform int in [lo, hi)."""
        return lo + self.randint(hi - lo)


def shuffle(seq, seed: int = 1):
    """Return a uniformly random permutation of `seq` (Fisher-Yates / Durstenfeld). Does not
    modify the input. Every one of the n! orderings is equally likely."""
    a = list(seq)
    rng = _Rng(seed)
    for i in range(len(a) - 1, 0, -1):
        j = rng.randint(i + 1)       # uniform in [0, i]
        a[i], a[j] = a[j], a[i]
    return a


def shuffle_inplace(a, rng: _Rng):
    """In-place Fisher-Yates using a supplied RNG (for callers that manage their own stream)."""
    for i in range(len(a) - 1, 0, -1):
        j = rng.randint(i + 1)
        a[i], a[j] = a[j], a[i]
    return a


def naive_shuffle(seq, seed: int = 1):
    """The BIASED naive shuffle: swap each position with a random position anywhere in the
    array. Included only to demonstrate the bias -- do not use it. Some permutations are more
    likely than others for n > 2."""
    a = list(seq)
    rng = _Rng(seed)
    n = len(a)
    for i in range(n):
        j = rng.randint(n)           # WRONG: full range, not [i, n)
        a[i], a[j] = a[j], a[i]
    return a


def sample_without_replacement(seq, k: int, seed: int = 1):
    """A uniform random k-subset of `seq` in random order, via a partial Fisher-Yates: only the
    first k swaps are needed. O(k), does not modify the input."""
    a = list(seq)
    n = len(a)
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    rng = _Rng(seed)
    for i in range(k):
        j = rng.randrange(i, n)
        a[i], a[j] = a[j], a[i]
    return a[:k]


def sattolo_cycle(seq, seed: int = 1):
    """Sattolo's algorithm: a uniform random CYCLIC permutation (a single n-cycle). Like
    Fisher-Yates but the swap partner is drawn from [0, i), strictly earlier, so the result is
    always one big cycle -- never a fixed point or a product of smaller cycles."""
    a = list(seq)
    rng = _Rng(seed)
    for i in range(len(a) - 1, 0, -1):
        j = rng.randint(i)           # uniform in [0, i) -- strictly less than i
        a[i], a[j] = a[j], a[i]
    return a


def is_permutation(original, shuffled) -> bool:
    """True if `shuffled` is a rearrangement of `original` (same multiset of elements)."""
    return sorted(original) == sorted(shuffled)


def is_single_cycle(perm) -> bool:
    """True if `perm` (a permutation of range(n) as a list) is a single n-cycle -- follow it
    from 0 and check every element is visited before returning."""
    n = len(perm)
    if n <= 1:
        return True
    seen = 0
    x = 0
    for _ in range(n):
        x = perm[x]
        seen += 1
        if x == 0:
            break
    return seen == n


def permutation_counts(n: int, trials: int, shuffler=shuffle, seed: int = 1):
    """Shuffle range(n) `trials` times and count how often each of the n! permutations appears.
    Returns {permutation_tuple: count}. Used to check uniformity."""
    counts = {}
    for t in range(trials):
        perm = tuple(shuffler(range(n), seed=seed + t))
        counts[perm] = counts.get(perm, 0) + 1
    return counts


def chi_square_uniform(counts, n_perms: int, trials: int) -> float:
    """Chi-square of observed permutation counts against the uniform expectation trials/n_perms.
    Missing permutations count as zero observations."""
    expected = trials / n_perms
    total = sum(counts.values())
    # include permutations that never appeared
    chi = 0.0
    for c in counts.values():
        chi += (c - expected) ** 2 / expected
    missing = n_perms - len(counts)
    chi += missing * (expected)      # (0 - expected)^2 / expected = expected, times missing
    return chi
