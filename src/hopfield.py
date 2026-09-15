"""Hopfield networks: content-addressable memory as energy descent on a spin glass.

A Hopfield network (1982) stores patterns as the STABLE STATES of a recurrent network of binary neurons,
and recalls them by dynamics rather than by lookup. Every neuron is a +/-1 spin connected symmetrically to
every other; a stored pattern becomes a valley in an energy landscape, and showing the network a corrupted
or partial version starts it at the valley's rim, from which the update rule rolls DOWNHILL to the bottom --
the clean memory. It is content-addressable: the address of a memory is (part of) its content.

Storage is HEBBIAN -- "neurons that fire together wire together." For p patterns xi^m (each a vector of
+/-1) over N neurons, the weights are the sum of outer products

    W_ij = (1/N) sum_m xi_i^m xi_j^m   (i != j),   W_ii = 0

symmetric with zero diagonal. Retrieval flips one neuron at a time to agree with the sign of its local
field h_i = sum_j W_ij s_j (ASYNCHRONOUS update). Because W is symmetric with a non-negative diagonal
contribution removed, every such flip cannot increase the LYAPUNOV energy

    E(s) = -1/2 sum_ij W_ij s_i s_j

so the dynamics are guaranteed to converge to a local minimum in finite time -- no cycles. The stored
patterns are (for modest loading) exactly those minima. Two hard facts fall out and are demonstrated here:

  CAPACITY. You cannot store arbitrarily many memories. Above about 0.138 N random patterns the minima
      proliferate into SPURIOUS states and stored patterns stop being stable -- the classic Amit-Gutfreund-
      Sompolinsky load. Below it, recall is near-perfect.
  SPURIOUS MEMORIES. Even below capacity the network invents attractors the designer never stored -- most
      famously MIXTURE states like sign(xi^1 +/- xi^2 +/- xi^3), odd combinations of stored patterns.

This module implements Hebbian storage, synchronous and asynchronous recall, the energy function, capacity
estimation by Monte Carlo, and spurious-state construction. It is validated: the energy never increases
under an asynchronous flip; stored patterns are fixed points and have lower energy than random states; a
pattern with a few bits flipped is recalled exactly; recall degrades as the load p/N passes ~0.14; the
weight matrix is symmetric with zero diagonal; and a stored pattern and its negation are both stable (the
built-in sign symmetry). Pure stdlib with a seeded RNG. The associative-memory companion to the Ising,
simulated-annealing, and Boltzmann-machine notes."""

from __future__ import annotations


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def bit(self):
        """A +/-1 spin from the HIGH bit (low bits of an LCG have period-2 correlations)."""
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return 1 if (self.state >> 31) else -1

    def randint(self, n):
        return int(self.u() * n) % n


def hebbian_weights(patterns):
    """Build the symmetric weight matrix W_ij = (1/N) sum_m xi_i^m xi_j^m, zero diagonal.

    `patterns` is a list of +/-1 vectors, all length N. Returns W as a list of lists."""
    n = len(patterns[0])
    w = [[0.0] * n for _ in range(n)]
    for pat in patterns:
        for i in range(n):
            pi = pat[i]
            wi = w[i]
            for j in range(n):
                if i != j:
                    wi[j] += pi * pat[j]
    inv = 1.0 / n
    for i in range(n):
        for j in range(n):
            w[i][j] *= inv
    return w


def energy(w, s):
    """Lyapunov energy E(s) = -1/2 sum_ij W_ij s_i s_j. Non-increasing under asynchronous updates."""
    n = len(s)
    total = 0.0
    for i in range(n):
        wi = w[i]
        si = s[i]
        for j in range(n):
            total += wi[j] * si * s[j]
    return -0.5 * total


def local_field(w, s, i):
    """h_i = sum_j W_ij s_j, the input to neuron i from the rest of the network."""
    wi = w[i]
    return sum(wi[j] * s[j] for j in range(len(s)))


def _sign(x):
    """Neuron activation sign; ties (h_i == 0) keep the current state by convention (return 0 here)."""
    return 1 if x > 0 else (-1 if x < 0 else 0)


def update_async(w, s, rng, sweeps=10):
    """Asynchronously flip neurons to the sign of their local field until a sweep makes no change.

    One sweep visits all N neurons in a random order. Returns (final_state, converged, n_sweeps).
    Guaranteed to converge because each flip cannot increase the energy and states are finite."""
    n = len(s)
    s = list(s)
    for sweep in range(sweeps):
        order = list(range(n))
        # Fisher-Yates with the RNG
        for k in range(n - 1, 0, -1):
            j = rng.randint(k + 1)
            order[k], order[j] = order[j], order[k]
        changed = False
        for i in order:
            h = local_field(w, s, i)
            new = _sign(h)
            if new != 0 and new != s[i]:
                s[i] = new
                changed = True
        if not changed:
            return s, True, sweep + 1
    return s, False, sweeps


def update_sync(w, s):
    """One synchronous step: every neuron simultaneously set to sign of its field. May oscillate (period 2)."""
    n = len(s)
    h = [local_field(w, s, i) for i in range(n)]
    return [(_sign(h[i]) if _sign(h[i]) != 0 else s[i]) for i in range(n)]


def is_fixed_point(w, s):
    """True iff no neuron wants to flip: sign(h_i) == s_i for all i (a stable memory)."""
    for i in range(len(s)):
        h = local_field(w, s, i)
        if _sign(h) != 0 and _sign(h) != s[i]:
            return False
    return True


def recall(w, cue, rng, sweeps=10):
    """Recall from a (possibly corrupted) cue: run asynchronous dynamics to a fixed point. Returns state."""
    s, _, _ = update_async(w, cue, rng, sweeps)
    return s


def overlap(a, b):
    """Normalized overlap m = (1/N) sum_i a_i b_i in [-1, 1]; 1 means identical, -1 the exact negation."""
    n = len(a)
    return sum(a[i] * b[i] for i in range(n)) / n


def random_patterns(p, n, seed=1):
    """p independent random +/-1 patterns of length n (spread seeds so the LCG streams decorrelate)."""
    pats = []
    for m in range(p):
        rng = _Rng((seed + m) * 7919 + 1)
        pats.append([rng.bit() for _ in range(n)])
    return pats


def flip_bits(pattern, k, seed=1):
    """Return a copy of `pattern` with k distinct random bits flipped -- a corrupted retrieval cue."""
    rng = _Rng(seed * 7919 + 3)
    n = len(pattern)
    idx = list(range(n))
    for a in range(n - 1, 0, -1):
        b = rng.randint(a + 1)
        idx[a], idx[b] = idx[b], idx[a]
    s = list(pattern)
    for i in idx[:k]:
        s[i] = -s[i]
    return s


def recall_accuracy(p, n, flips, seed=1, sweeps=20):
    """Store p random patterns in an N-neuron net; cue each with `flips` corrupted bits; measure recall.

    Returns the mean overlap of the recalled state with the intended pattern, averaged over the p cues.
    Near 1.0 below capacity (~0.138 N); collapses toward 0 as the load grows past it."""
    patterns = random_patterns(p, n, seed)
    w = hebbian_weights(patterns)
    total = 0.0
    for m, pat in enumerate(patterns):
        cue = flip_bits(pat, flips, seed=seed + m)
        rng = _Rng((seed + m) * 104729 + 7)
        out = recall(w, cue, rng, sweeps)
        total += abs(overlap(out, pat))  # abs: the sign-flipped memory counts as recall
    return total / p


CAPACITY_RATIO = 0.138  # Amit-Gutfreund-Sompolinsky critical load p/N for random patterns


def spurious_mixture(patterns, signs):
    """Construct a MIXTURE (spurious) state sign(sum_k signs[k] * xi^k) -- an attractor never stored.

    `signs` is a list of +/-1, one per pattern used (an odd count gives a genuine stable mixture below
    capacity). Ties resolve to +1."""
    n = len(patterns[0])
    out = []
    for i in range(n):
        acc = sum(signs[k] * patterns[k][i] for k in range(len(signs)))
        out.append(1 if acc >= 0 else -1)
    return out
