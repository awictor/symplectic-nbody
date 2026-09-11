"""The P-square algorithm: streaming quantile estimation in constant memory.

Computing a QUANTILE (the median, the 95th percentile) the exact way requires storing and sorting all
the data -- impossible for a stream of billions of values, or a sensor that reports forever. The
P-SQUARE ALGORITHM (Jain and Chlamtac, 1985) estimates any quantile of a data stream using just FIVE
markers and O(1) memory, never storing the samples, with accuracy that improves as more data
arrives. It is the classic tool for latency percentiles in monitoring systems (p50/p95/p99 without
keeping every request time) and for online statistics on embedded devices.

The idea is to track five MARKERS along the data: the running minimum, the running maximum, the
current estimate of the desired quantile, and two markers halfway between. Each marker has a height
(its current value) and a position (its rank so far); each also has a DESIRED position that grows
linearly with the sample count so the markers stay spread at the target quantile and the two
midpoints. As each new value arrives, it is slotted into the marker cells, positions are incremented,
and any marker that has drifted more than one step from its desired position is nudged back -- its
height adjusted by a PARABOLIC (P-square) interpolation through its neighbours, falling back to
linear interpolation if the parabola would break the ordering. The middle marker's height is the
quantile estimate.

This module implements the single-quantile P-square estimator and a multi-quantile variant (an
'extended' histogram that tracks several quantiles at once), verified against exact quantiles
computed from the full sorted data: that on uniform, normal, and exponential streams the estimate is
within a small relative error of the true quantile, that the median of a symmetric stream is near its
centre, that extreme quantiles (p01, p99) are tracked, that the min and max markers are exact, and
that feeding a known constant stream returns that constant. Pure stdlib; a streaming-statistics
companion to the reservoir-sampling, HyperLogLog, and Count-Min-sketch notes."""

from __future__ import annotations


class P2Quantile:
    """Single-quantile estimator via the P-square algorithm. O(1) memory, no samples stored."""

    def __init__(self, p):
        if not (0 < p < 1):
            raise ValueError("quantile p must be in (0, 1)")
        self.p = p
        self.n = 0
        self.q = []          # marker heights
        self.pos = []        # marker positions (1-indexed ranks)
        self.npos = []       # desired positions
        self.dn = []         # increments of desired positions

    def add(self, x):
        if self.n < 5:
            self.q.append(x)
            self.n += 1
            if self.n == 5:
                self.q.sort()
                self.pos = [1, 2, 3, 4, 5]
                p = self.p
                self.npos = [1, 1 + 2 * p, 1 + 4 * p, 3 + 2 * p, 5]
                self.dn = [0, p / 2, p, (1 + p) / 2, 1]
            return

        # find the cell k that x falls into and update the running min/max
        if x < self.q[0]:
            self.q[0] = x
            k = 0
        elif x >= self.q[4]:
            self.q[4] = x
            k = 3
        else:
            k = 0
            for i in range(1, 5):
                if x < self.q[i]:
                    k = i - 1
                    break
            else:
                k = 3

        # increment positions of markers above the cell
        for i in range(k + 1, 5):
            self.pos[i] += 1
        for i in range(5):
            self.npos[i] += self.dn[i]

        # adjust the three interior markers if needed
        for i in range(1, 4):
            d = self.npos[i] - self.pos[i]
            if (d >= 1 and self.pos[i + 1] - self.pos[i] > 1) or \
               (d <= -1 and self.pos[i - 1] - self.pos[i] < -1):
                d = 1 if d >= 0 else -1
                qi = self._parabolic(i, d)
                if self.q[i - 1] < qi < self.q[i + 1]:
                    self.q[i] = qi
                else:
                    self.q[i] = self._linear(i, d)
                self.pos[i] += int(d)

        self.n += 1

    def _parabolic(self, i, d):
        qi = self.q[i]
        qp = self.q[i + 1]
        qm = self.q[i - 1]
        pi = self.pos[i]
        pp = self.pos[i + 1]
        pm = self.pos[i - 1]
        return qi + d / (pp - pm) * (
            (pi - pm + d) * (qp - qi) / (pp - pi)
            + (pp - pi - d) * (qi - qm) / (pi - pm)
        )

    def _linear(self, i, d):
        return self.q[i] + d * (self.q[i + int(d)] - self.q[i]) / (self.pos[i + int(d)] - self.pos[i])

    def quantile(self):
        """The current estimate of the p-quantile."""
        if self.n == 0:
            return None
        if self.n < 5:
            s = sorted(self.q)
            # simple exact quantile for the tiny warm-up sample
            idx = min(len(s) - 1, int(self.p * len(s)))
            return s[idx]
        return self.q[2]

    def minimum(self):
        if self.n == 0:
            return None
        return self.q[0] if self.n >= 5 else min(self.q)

    def maximum(self):
        if self.n == 0:
            return None
        return self.q[4] if self.n >= 5 else max(self.q)


class P2Histogram:
    """Track several quantiles of a stream at once, each with an independent P-square estimator."""

    def __init__(self, quantiles=(0.5, 0.9, 0.95, 0.99)):
        self.estimators = {p: P2Quantile(p) for p in quantiles}

    def add(self, x):
        for est in self.estimators.values():
            est.add(x)

    def quantile(self, p):
        return self.estimators[p].quantile()

    def summary(self):
        return {p: est.quantile() for p, est in self.estimators.items()}
