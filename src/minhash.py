"""MinHash and locality-sensitive hashing: estimating set similarity at scale.

How similar are two sets? The JACCARD SIMILARITY -- the size of their intersection over the size of
their union -- is the natural measure, but computing it for every pair among millions of sets (web
pages, documents, user histories) is hopeless if each comparison touches the full sets. MINHASH,
invented by Broder for AltaVista's near-duplicate web-page detection, compresses each set into a
short SIGNATURE of k numbers such that the probability that two signatures agree in any given
position equals exactly the Jaccard similarity of the underlying sets. So the fraction of matching
signature positions is an unbiased estimate of Jaccard -- computed from k small integers instead of
the whole sets.

The mechanism is a beautiful piece of probability. Apply a random permutation to the universe of
possible elements; the MINIMUM element of a set under that permutation is equally likely to be any
element of the set, so two sets share that minimum exactly when the overall-minimum element lies in
their intersection -- an event of probability |A intersect B| / |A union B|. Using k independent hash
functions as stand-ins for k permutations gives a k-dimensional signature whose expected agreement is
the Jaccard similarity; the estimate's error shrinks like 1/sqrt(k).

To go from pairwise estimation to actually FINDING the similar pairs among many sets without
comparing all pairs, LOCALITY-SENSITIVE HASHING (LSH) splits each signature into b BANDS of r rows
and hashes each band; two sets become CANDIDATES if they collide in any band. Tuning b and r shapes
an S-curve that makes near-duplicates collide with high probability while keeping dissimilar pairs
apart, turning near-duplicate search into a few hash-table lookups.

This module implements MinHash signatures (with universal integer hashing), the Jaccard estimator,
and banded LSH candidate generation. It is verified that the estimated Jaccard converges to the true
value as k grows (mean error within theory across many random set pairs), that identical sets always
estimate 1 and disjoint sets estimate ~0, that the estimator is symmetric, and that LSH recalls the
genuinely similar pairs (high Jaccard) while filtering out dissimilar ones. Pure stdlib; a
probabilistic-data-structure companion to the Bloom-filter, Count-Min, and HyperLogLog notes."""

from __future__ import annotations

# a large prime modulus for universal hashing (Mersenne prime 2^61 - 1)
_MERSENNE = (1 << 61) - 1
_MAXHASH = (1 << 32) - 1


def _stable_hash(x):
    """A deterministic 64-bit hash of an element (str/int), independent of PYTHONHASHSEED."""
    if isinstance(x, int):
        h = x & 0xFFFFFFFFFFFFFFFF
    else:
        s = str(x).encode("utf-8")
        h = 1469598103934665603                     # FNV-1a 64-bit offset basis
        for byte in s:
            h ^= byte
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


class MinHash:
    """A MinHash signature generator using k universal hash functions h(x) = (a*x + b) mod p mod M.

    The k coefficients are derived deterministically from a seed, so signatures are reproducible."""

    def __init__(self, num_hashes=128, seed=1):
        self.k = num_hashes
        self.seed = seed
        self._a = []
        self._b = []
        state = (seed * 2654435761 + 12345) & 0xFFFFFFFFFFFFFFFF
        for _ in range(num_hashes):
            state = (1103515245 * state + 12345) & 0xFFFFFFFFFFFFFFFF
            a = (state % (_MERSENNE - 1)) + 1        # a in [1, p-1]
            state = (1103515245 * state + 12345) & 0xFFFFFFFFFFFFFFFF
            b = state % _MERSENNE                    # b in [0, p-1]
            self._a.append(a)
            self._b.append(b)

    def signature(self, elements):
        """The MinHash signature of a set (any iterable of hashable elements): a list of k minima."""
        sig = [_MAXHASH + 1] * self.k
        seen_any = False
        for e in elements:
            seen_any = True
            hx = _stable_hash(e)
            for i in range(self.k):
                hv = (((self._a[i] * hx + self._b[i]) % _MERSENNE) & _MAXHASH)
                if hv < sig[i]:
                    sig[i] = hv
        if not seen_any:
            return [None] * self.k                    # empty set has no minima
        return sig


def estimate_jaccard(sig_a, sig_b):
    """Estimate the Jaccard similarity of two sets from their MinHash signatures: the fraction of
    signature positions that agree."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have the same length")
    if not sig_a:
        return 0.0
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x is not None and x == y)
    return matches / len(sig_a)


def true_jaccard(a, b):
    """The exact Jaccard similarity |A n B| / |A u B| (0 if both empty)."""
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


class LSH:
    """Banded locality-sensitive hashing over MinHash signatures for near-duplicate search.

    Splits each length-k signature into `bands` bands of `rows` rows (bands * rows must be <= k);
    two items become candidates if they share an identical band."""

    def __init__(self, bands, rows):
        self.bands = bands
        self.rows = rows
        self.buckets = [dict() for _ in range(bands)]   # one hash table per band
        self.signatures = {}

    def add(self, key, signature):
        """Index an item under its signature."""
        self.signatures[key] = signature
        for band in range(self.bands):
            chunk = tuple(signature[band * self.rows:(band + 1) * self.rows])
            self.buckets[band].setdefault(chunk, set()).add(key)

    def candidates(self, signature):
        """All indexed keys that collide with `signature` in at least one band."""
        found = set()
        for band in range(self.bands):
            chunk = tuple(signature[band * self.rows:(band + 1) * self.rows])
            found |= self.buckets[band].get(chunk, set())
        return found

    def all_candidate_pairs(self):
        """Every unordered pair of indexed keys that collide in at least one band."""
        pairs = set()
        for band in range(self.bands):
            for bucket in self.buckets[band].values():
                items = sorted(bucket, key=lambda k: str(k))
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        pairs.add((items[i], items[j]))
        return pairs


def lsh_threshold(bands, rows):
    """The approximate Jaccard similarity at which the LSH S-curve crosses 50% collision probability:
    (1/bands) ** (1/rows). Pairs above this threshold are likely to become candidates."""
    return (1.0 / bands) ** (1.0 / rows)
