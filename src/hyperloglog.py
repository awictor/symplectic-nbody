"""HyperLogLog: counting millions of distinct items in kilobytes.

How many DISTINCT visitors, IP addresses, or search terms appeared in a stream? Counting exactly
means remembering every item you have seen -- linear memory. HyperLogLog (Flajolet et al., 2007)
estimates the cardinality to within a percent or two using a fixed, tiny amount of memory, no
matter how many items pass by. A billion distinct items fit in about 1.5 KB.

The intuition is a bet. Hash each item to a random bit string; the more distinct items you see,
the longer the longest run of leading zeros you are likely to encounter (seeing k leading zeros
has probability 2^-(k+1), so it suggests roughly 2^k distinct items). One such observation is
noisy, so HyperLogLog splits the hash: the first p bits pick one of m = 2^p registers, and each
register keeps the maximum leading-zero-count (+1) seen in its bucket. Averaging the 2^register
values across buckets -- with the HARMONIC mean, which tames the outliers -- gives

    E = alpha_m * m^2 / sum_j 2^{-M[j]},

and the relative error is about 1.04 / sqrt(m), so 1024 registers (1 KB) already give ~3%. Small
and huge cardinalities get bias corrections (linear counting when many registers are still
empty). Two sketches merge by taking the register-wise max, so counts are trivially parallel and
distributed -- which is why every large-scale analytics system (Redis, Presto, BigQuery) ships
it.

This module implements the registers, the add/estimate/merge operations, the small- and
large-range corrections, and the theoretical standard error, and checks the estimate against
true cardinalities. Pure stdlib; the streaming-cardinality companion to the Bloom-filter note.
"""

from __future__ import annotations

import math


def _fnv1a64(data: bytes) -> int:
    """64-bit FNV-1a hash of a byte string, with a final avalanche mix.

    Plain FNV-1a has weak diffusion in its high bits, which HyperLogLog reads to pick a
    register -- leaving many buckets never selected. A splitmix64-style finalizer (xor-shift and
    odd-constant multiplies) scrambles all 64 bits so the top p bits are uniform."""
    h = 0xcbf29ce484222325
    prime = 0x100000001b3
    mask = (1 << 64) - 1
    for byte in data:
        h ^= byte
        h = (h * prime) & mask
    # splitmix64 finalizer
    h = (h ^ (h >> 30)) & mask
    h = (h * 0xbf58476d1ce4e5b9) & mask
    h = (h ^ (h >> 27)) & mask
    h = (h * 0x94d049bb133111eb) & mask
    h = (h ^ (h >> 31)) & mask
    return h


def _alpha(m: int) -> float:
    """The bias-correction constant alpha_m for m registers."""
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / m)


class HyperLogLog:
    """A HyperLogLog sketch with 2^p registers (p in 4..16)."""

    def __init__(self, p: int = 12):
        if not 4 <= p <= 16:
            raise ValueError("p must be in 4..16")
        self.p = p
        self.m = 1 << p
        self.registers = bytearray(self.m)
        self.hash_bits = 64

    def add(self, item):
        """Add an item to the sketch."""
        data = item if isinstance(item, bytes) else str(item).encode("utf-8")
        x = _fnv1a64(data)
        idx = x >> (self.hash_bits - self.p)          # first p bits pick the register
        rest = (x << self.p) & ((1 << self.hash_bits) - 1)  # remaining bits (shifted up)
        rank = self._leading_zeros(rest, self.hash_bits) + 1
        if rank > self.registers[idx]:
            self.registers[idx] = rank

    @staticmethod
    def _leading_zeros(x: int, width: int) -> int:
        """Number of leading zero bits in the low `width` bits of x."""
        if x == 0:
            return width
        return width - x.bit_length()

    def estimate(self) -> float:
        """Estimate the number of distinct items added, with small/large-range corrections."""
        m = self.m
        # raw harmonic-mean estimate
        inv_sum = sum(2.0 ** (-r) for r in self.registers)
        raw = _alpha(m) * m * m / inv_sum
        # small-range correction: linear counting when registers are still empty
        if raw <= 2.5 * m:
            zeros = self.registers.count(0)
            if zeros != 0:
                return m * math.log(m / zeros)
        # large-range correction near the 2^64 hash ceiling (negligible for 64-bit hashes)
        two64 = 2.0 ** 64
        if raw > two64 / 30.0:
            return -two64 * math.log(1.0 - raw / two64)
        return raw

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        """Merge another sketch (same p) by register-wise max. Returns a new sketch representing
        the union of the two item sets."""
        if self.p != other.p:
            raise ValueError("cannot merge sketches with different precision")
        out = HyperLogLog(self.p)
        for i in range(self.m):
            out.registers[i] = max(self.registers[i], other.registers[i])
        return out

    def standard_error(self) -> float:
        """The theoretical relative standard error 1.04 / sqrt(m)."""
        return 1.04 / math.sqrt(self.m)


def relative_error(estimate: float, truth: int) -> float:
    """Signed relative error (estimate - truth) / truth."""
    if truth == 0:
        return 0.0 if estimate == 0 else float("inf")
    return (estimate - truth) / truth
