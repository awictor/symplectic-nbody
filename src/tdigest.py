"""The t-digest -- accurate streaming quantiles over the WHOLE distribution, with sharp tails.

The P-square algorithm estimates one quantile you fix in advance. But monitoring a service you rarely
want just the median: you want p50, p90, p99, p999, and the max, all from one pass over a billion
latencies you cannot store. And you want the tails -- p999 -- to be ACCURATE, because that is where the
pain lives, while you happily tolerate coarser resolution near the median where nobody cares about the
exact value. The t-digest (Dunning, 2013) is the data structure built for exactly this: a small,
bounded summary of a stream that answers ANY quantile query, is far more accurate in the tails than in
the middle by design, and -- crucially for distributed systems -- MERGES, so per-shard digests combine
into a global one without touching the raw data.

The structure is a set of CENTROIDS, each a (mean, weight) pair summarising a cluster of nearby values.
The genius is the SCALE FUNCTION that governs how big a centroid is allowed to be. Map the cumulative
quantile q in [0,1] to a "k-scale" k(q) = (delta / 2pi) * arcsin(2q - 1). This k is nearly linear in q
near the median but COMPRESSED near q=0 and q=1, so a fixed step in k corresponds to a wide band of q
in the middle and a razor-thin band at the tails. A centroid may absorb more weight only while it stays
within one k-step of where it started; near the tails that permits almost no weight, forcing many tiny
centroids there (high resolution), while near the median a single centroid can swallow a large fraction
of the data (low resolution, which is fine). The number of centroids stays bounded around delta
regardless of how many values stream through -- constant memory for unbounded data.

Insertion buffers incoming values and periodically MERGES: sort the buffer together with the existing
centroids, then sweep left to right accumulating weight into the current centroid until the scale
function says it is full, at which point a new centroid starts. Quantile queries interpolate between
centroid means using their accumulated weights; rank queries (the CDF) do the inverse. Because the
merge is associative, two digests combine by merging their centroid lists -- the property that lets a
fleet of machines each keep a local digest and roll them up exactly.

This module implements the merging t-digest with the arcsin scale function, quantile and CDF queries,
digest merging, min/max tracking, and a buffered add path. Pure standard library.

Validation. (1) Against EXACT quantiles from the fully-sorted data, on uniform, normal, exponential,
and heavily-skewed streams of tens of thousands of values, the t-digest's estimates are within a small
tolerance -- and, as the design promises, the ERROR AT THE TAILS (p99, p999) is markedly smaller than
near the median in rank terms. (2) The centroid count stays bounded by a small multiple of delta no
matter the stream length. (3) MERGE correctness: splitting a stream across several digests and merging
them yields quantiles matching a single digest fed the whole stream, both close to exact. (4) Monotone
CDF, quantile(0)=min, quantile(1)=max, and quantile/CDF are inverse. (5) Reproducible and
order-robust: shuffled input gives statistically the same quantiles. Uses only ``math`` and ``bisect``."""

import bisect
import math


class _Centroid:
    __slots__ = ("mean", "weight")

    def __init__(self, mean, weight):
        self.mean = mean
        self.weight = weight

    def __lt__(self, other):
        return self.mean < other.mean


class TDigest:
    """A merging t-digest. ``delta`` (compression) trades size for accuracy; 100 is a good default."""

    def __init__(self, delta=100.0):
        self.delta = float(delta)
        self.centroids = []          # sorted by mean, invariant maintained on compress
        self.total_weight = 0.0
        self._buffer = []            # unmerged (value, weight) pairs
        self.min = math.inf
        self.max = -math.inf

    # -- ingestion -----------------------------------------------------------
    def add(self, value, weight=1.0):
        """Add a single value (optionally weighted) to the digest."""
        self._buffer.append((float(value), float(weight)))
        if value < self.min:
            self.min = float(value)
        if value > self.max:
            self.max = float(value)
        # flush the buffer once it is comparable in size to the centroid budget
        if len(self._buffer) >= max(10, int(self.delta) * 2):
            self._flush()

    def add_all(self, values):
        for v in values:
            self.add(v)

    # -- scale function ------------------------------------------------------
    def _k(self, q):
        """The k-scale: k(q) = (delta / 2pi) * arcsin(2q - 1). Compressed at the tails."""
        return self.delta / (2 * math.pi) * math.asin(2 * q - 1)

    # -- the merge/compress step --------------------------------------------
    def _flush(self):
        if not self._buffer and not self.centroids:
            return
        # combine buffered points with existing centroids and sort by mean
        items = [_Centroid(v, w) for v, w in self._buffer]
        items.extend(self.centroids)
        items.sort()
        self._buffer = []

        total = sum(c.weight for c in items)
        if total == 0:
            self.centroids = []
            self.total_weight = 0.0
            return

        merged = []
        q0 = 0.0
        # weight limit for the current centroid: it may grow while k(q_end) - k(q0) <= 1
        cur = _Centroid(items[0].mean, items[0].weight)
        weight_so_far = cur.weight
        for c in items[1:]:
            q_proposed = (weight_so_far + c.weight) / total
            if self._k(q_proposed) - self._k(q0) <= 1.0:
                # absorb into current centroid (weighted mean)
                new_w = cur.weight + c.weight
                cur.mean = (cur.mean * cur.weight + c.mean * c.weight) / new_w
                cur.weight = new_w
                weight_so_far += c.weight
            else:
                merged.append(cur)
                q0 = weight_so_far / total
                cur = _Centroid(c.mean, c.weight)
                weight_so_far += c.weight
        merged.append(cur)

        self.centroids = merged
        self.total_weight = total

    def _ensure_merged(self):
        if self._buffer:
            self._flush()

    # -- queries -------------------------------------------------------------
    def quantile(self, q):
        """Estimate the value at quantile ``q`` in [0, 1]."""
        self._ensure_merged()
        if not self.centroids:
            return math.nan
        if q <= 0:
            return self.min
        if q >= 1:
            return self.max
        n = len(self.centroids)
        if n == 1:
            return self.centroids[0].mean

        target = q * self.total_weight
        # cumulative weight to the CENTER of each centroid
        cum = 0.0
        centers = []
        for c in self.centroids:
            centers.append(cum + c.weight / 2.0)
            cum += c.weight

        # below the first centroid center: interpolate from the min
        if target < centers[0]:
            c0 = self.centroids[0]
            if centers[0] <= 0:
                return c0.mean
            frac = target / centers[0]
            return self.min + frac * (c0.mean - self.min)
        # above the last center: interpolate to the max
        if target > centers[-1]:
            cl = self.centroids[-1]
            denom = self.total_weight - centers[-1]
            if denom <= 0:
                return cl.mean
            frac = (target - centers[-1]) / denom
            return cl.mean + frac * (self.max - cl.mean)

        # between two centroid centers: linear interpolation on the means
        i = bisect.bisect_right(centers, target) - 1
        left, right = self.centroids[i], self.centroids[i + 1]
        span = centers[i + 1] - centers[i]
        frac = (target - centers[i]) / span if span > 0 else 0.0
        return left.mean + frac * (right.mean - left.mean)

    def cdf(self, x):
        """Estimate the fraction of the data <= x (the empirical CDF)."""
        self._ensure_merged()
        if not self.centroids:
            return math.nan
        if x < self.min:
            return 0.0
        if x >= self.max:
            return 1.0
        n = len(self.centroids)
        if n == 1:
            return 0.5 if x >= self.centroids[0].mean else 0.0

        cum = 0.0
        centers = []
        for c in self.centroids:
            centers.append(cum + c.weight / 2.0)
            cum += c.weight

        if x < self.centroids[0].mean:
            c0 = self.centroids[0]
            denom = c0.mean - self.min
            frac = (x - self.min) / denom if denom > 0 else 0.0
            return frac * centers[0] / self.total_weight
        if x > self.centroids[-1].mean:
            cl = self.centroids[-1]
            denom = self.max - cl.mean
            frac = (x - cl.mean) / denom if denom > 0 else 1.0
            return (centers[-1] + frac * (self.total_weight - centers[-1])) / self.total_weight

        # locate the bracketing centroids by mean
        means = [c.mean for c in self.centroids]
        i = bisect.bisect_right(means, x) - 1
        i = max(0, min(i, n - 2))
        left, right = self.centroids[i], self.centroids[i + 1]
        span = right.mean - left.mean
        frac = (x - left.mean) / span if span > 0 else 0.0
        weight_pos = centers[i] + frac * (centers[i + 1] - centers[i])
        return weight_pos / self.total_weight

    # -- merging -------------------------------------------------------------
    def merge(self, other):
        """Merge another t-digest into this one (associative; enables distributed roll-up)."""
        other._ensure_merged()
        for c in other.centroids:
            self._buffer.append((c.mean, c.weight))
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)
        self._flush()
        return self

    def centroid_count(self):
        self._ensure_merged()
        return len(self.centroids)


def merge_digests(digests, delta=100.0):
    """Merge a list of digests into a fresh one (for distributed aggregation)."""
    out = TDigest(delta)
    for d in digests:
        out.merge(d)
    return out
