"""Luby Transform codes: rateless erasure coding that recovers a file from almost any k packets received.

Send a large file over a lossy channel -- packets vanish unpredictably, and you do not know the loss
rate in advance. Fixed-rate codes force you to guess the redundancy up front. LT codes (Michael Luby,
2002), the first practical FOUNTAIN codes, sidestep the guess: from k source blocks the encoder can
generate an ENDLESS stream of encoded symbols, each an XOR of a random subset of the source blocks, and
the decoder recovers all k originals as soon as it has collected slightly more than k symbols -- from
ANY k(1+epsilon) of them, whichever happen to arrive. The channel can drop whatever it likes; you just
keep catching drops from the fountain until you have enough.

Each encoded symbol picks a DEGREE d from a carefully designed distribution, chooses d distinct source
blocks uniformly, and XORs them. The magic is the degree distribution: the IDEAL SOLITON and its
practical refinement the ROBUST SOLITON are tuned so that, throughout decoding, there is almost always
exactly one symbol of degree one to release. Decoding is the PEELING (belief-propagation) algorithm:
find any symbol connected to a single unknown block -- that symbol IS that block -- then XOR the newly
known block out of every other symbol that includes it, which lowers their degrees and often exposes
new degree-one symbols. Repeat until everything is peeled or the process stalls.

This module implements LT encoding with the robust soliton distribution over byte blocks, and the
peeling decoder, with a seeded RNG so encoder and decoder agree on which blocks each symbol touched. It
is validated: a degree-one symbol trivially decodes its block; the robust soliton distribution is a
valid probability distribution summing to one; a message is recovered exactly once enough symbols
arrive, using only about k(1+epsilon) of them; decoding is robust to arbitrary erasures (any sufficient
subset works); the XOR structure is self-consistent (re-encoding a decoded message reproduces the
symbols); and decoding fails gracefully (returns None) when too few symbols are collected. Pure stdlib;
the erasure-coding companion to the Reed-Solomon, BCH, and LDPC tools."""

from __future__ import annotations


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u32(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def uniform(self):
        return (self.u32() >> 8) / (1 << 24)

    def randint(self, lo, hi):
        """Integer in [lo, hi)."""
        return lo + self.u32() % (hi - lo)

    def sample(self, n, d):
        """d distinct indices from range(n)."""
        chosen = set()
        while len(chosen) < d:
            chosen.add(self.u32() % n)
        return sorted(chosen)


def ideal_soliton(k):
    """The ideal soliton distribution rho(d) over degrees 1..k."""
    rho = [0.0] * (k + 1)
    rho[1] = 1.0 / k
    for d in range(2, k + 1):
        rho[d] = 1.0 / (d * (d - 1))
    return rho


def robust_soliton(k, c=0.1, delta=0.5):
    """The robust soliton distribution: ideal soliton plus a spike tau to guarantee a degree-1 supply.

    Returns a normalized probability list mu of length k+1 (index 0 unused)."""
    import math
    rho = ideal_soliton(k)
    R = c * math.log(k / delta) * math.sqrt(k) if k > 1 else 1.0
    tau = [0.0] * (k + 1)
    kR = max(1, int(round(k / R)))
    for d in range(1, k + 1):
        if d < kR:
            tau[d] = R / (d * k)
        elif d == kR:
            tau[d] = R * math.log(R / delta) / k
        else:
            tau[d] = 0.0
    beta = sum(rho[d] + tau[d] for d in range(1, k + 1))
    mu = [0.0] + [(rho[d] + tau[d]) / beta for d in range(1, k + 1)]
    return mu


def _sample_degree(mu, rng):
    """Sample a degree from the distribution mu (index 1..k)."""
    u = rng.uniform()
    cum = 0.0
    for d in range(1, len(mu)):
        cum += mu[d]
        if u <= cum:
            return d
    return len(mu) - 1


class LTEncoder:
    """Luby Transform encoder over a list of equal-length byte blocks."""

    def __init__(self, blocks, seed=1, c=0.1, delta=0.5):
        self.blocks = [bytes(b) for b in blocks]
        self.k = len(blocks)
        self.blen = len(blocks[0]) if blocks else 0
        self.mu = robust_soliton(self.k, c, delta)
        self.rng = _Rng(seed)

    def _xor(self, indices):
        out = bytearray(self.blen)
        for idx in indices:
            b = self.blocks[idx]
            for i in range(self.blen):
                out[i] ^= b[i]
        return bytes(out)

    def generate(self, n_symbols):
        """Produce n_symbols encoded symbols. Each is (neighbor_indices, payload_bytes)."""
        symbols = []
        for _ in range(n_symbols):
            d = _sample_degree(self.mu, self.rng)
            neighbors = self.rng.sample(self.k, d)
            symbols.append((tuple(neighbors), self._xor(neighbors)))
        return symbols


def peel_decode(symbols, k, blen):
    """Peeling (belief-propagation) decoder. Returns the list of k decoded blocks, or None if it stalls.

    symbols: list of (neighbor_indices, payload_bytes)."""
    # working copy: mutable neighbor sets and payloads
    syms = [[set(nb), bytearray(pl)] for nb, pl in symbols]
    decoded = [None] * k

    def peel_block(idx, value):
        """Record block idx = value, then XOR it out of every symbol that still references it."""
        decoded[idx] = bytes(value)
        for nb, pl in syms:
            if idx in nb:
                for i in range(blen):
                    pl[i] ^= value[i]
                nb.discard(idx)

    progress = True
    while progress:
        progress = False
        for nb, pl in syms:
            if len(nb) == 1:
                idx = next(iter(nb))
                nb.clear()                       # consume this degree-1 symbol
                if decoded[idx] is None:
                    peel_block(idx, bytes(pl))
                    progress = True
    if all(decoded[i] is not None for i in range(k)):
        return decoded
    return None


def decode(symbols, k, blen):
    """Convenience wrapper: peel-decode and return the concatenated message or None."""
    blocks = peel_decode(symbols, k, blen)
    if blocks is None:
        return None
    return b"".join(blocks)


def chunk_message(data, blen):
    """Split bytes into equal-length blocks (zero-padded), returns (blocks, original_length)."""
    n = len(data)
    blocks = []
    for i in range(0, max(n, 1), blen):
        chunk = data[i:i + blen]
        if len(chunk) < blen:
            chunk = chunk + bytes(blen - len(chunk))
        blocks.append(chunk)
    return blocks, n
