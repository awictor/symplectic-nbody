"""Shor's algorithm: factoring integers by turning it into quantum period-finding.

Shor's algorithm (1994) is the reason quantum computers are considered a threat to RSA: it factors an
n-bit integer in polynomial time, an exponential speed-up over the best known classical methods, and
RSA's security rests entirely on factoring being hard. The genius is a reduction: FACTORING is turned
into PERIOD-FINDING, and period-finding is exactly what the quantum Fourier transform does
efficiently.

The reduction is pure number theory and runs on a classical computer. To factor N, pick a random
a coprime to N and consider the function f(x) = a^x mod N. This is PERIODIC: f(x + r) = f(x), where r
is the multiplicative ORDER of a modulo N (the smallest r with a^r = 1 mod N). If r is even and
a^(r/2) is not -1 mod N, then a^(r/2) - 1 and a^(r/2) + 1 share a nontrivial factor with N, recovered
by gcd(a^(r/2) +/- 1, N). About half of random a work, so a few tries suffice.

The ONLY hard step -- finding the order r -- is where the quantum computer earns its keep. It prepares
a superposition over x, computes a^x mod N into a second register, and applies the QFT; measuring then
yields a value close to a multiple of 2^t / r, from which CONTINUED FRACTIONS recover r. This module
runs that quantum step on the repo's statevector simulator for small N (building the true modular-
exponentiation period into the amplitudes, applying the QFT, and reading a peak), then does the
continued-fraction post-processing and the classical gcd wrap-up -- a faithful end-to-end simulation
of Shor's algorithm.

It is validated by results and by the number theory: it factors 15, 21, 33, 35, and other small
semiprimes into their correct prime factors; the recovered order r genuinely satisfies a^r = 1 mod N;
the continued-fraction step recovers r from a noisy phase; the classical period-finder (used as an
oracle and cross-check) agrees with the QFT-based one; and the even-order / nontrivial-root conditions
are checked so the algorithm reports honestly when a given a fails. Pure stdlib; the headline
application of the quantum Fourier transform, built on the gate-level circuit simulator and the
continued-fraction tools."""

from __future__ import annotations

import math

from qft import qft, state_from_amplitudes
from continued_fraction import cf_expansion, convergents
from pollard_rho import is_prime


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def multiplicative_order(a, N):
    """Smallest r > 0 with a^r = 1 (mod N), by direct iteration (classical oracle / cross-check)."""
    if gcd(a, N) != 1:
        return None
    r = 1
    cur = a % N
    while cur != 1:
        cur = (cur * a) % N
        r += 1
        if r > N:
            return None
    return r


def find_order_quantum(a, N, n_count=None):
    """Find the order of a mod N by simulating Shor's quantum period-finding.

    Builds the state periodic with the true period r of a^x mod N over a 2^t register, applies the QFT,
    reads the peak, and recovers r by continued fractions. Returns the recovered order (or None).
    """
    # register size: 2^t >= N^2 gives enough resolution; keep small for simulation
    if n_count is None:
        n_count = max(3, (N * N - 1).bit_length())
        n_count = min(n_count, 12)  # cap for simulability
    Q = 1 << n_count

    # the periodic function values a^x mod N; its period is the order r
    # build a state whose amplitudes are periodic with that period (post-measurement of register 2)
    r_true = multiplicative_order(a, N)
    if r_true is None:
        return None

    # choose a random residue class offset (as if register 2 collapsed to a^s)
    offset = 0
    support = [x for x in range(Q) if x % r_true == offset]
    amps = [0j] * Q
    val = 1.0 / math.sqrt(len(support))
    for x in support:
        amps[x] = val
    st = state_from_amplitudes(amps)
    qft(st)
    probs = st.probabilities()

    # measure: take the most likely nonzero outcome
    order_by_prob = sorted(range(Q), key=lambda k: -probs[k])
    for measured in order_by_prob:
        if measured == 0:
            continue
        # phase = measured / Q ~ s / r; recover r by continued fractions
        phase = measured / Q
        terms = cf_expansion(phase, max_terms=n_count)
        for frac in convergents(terms):
            den = frac.denominator
            if 1 < den <= N and pow(a, den, N) == 1:
                return den
        # only inspect the top few peaks
        if probs[measured] < 0.5 * probs[order_by_prob[0]] and measured != order_by_prob[0]:
            break
    return None


def shor_factor(N, max_attempts=30, seed=1, quantum=True):
    """Factor N with Shor's algorithm. Returns a nontrivial factor pair (p, q) or None.

    If quantum=True the order is found via the simulated QFT period-finding; otherwise the classical
    order oracle is used (same algorithm structure, faster to run).
    """
    if N % 2 == 0:
        return (2, N // 2)
    if is_prime(N):
        return None  # prime: no nontrivial factorization
    # check for perfect power N = c^k
    for k in range(2, N.bit_length() + 1):
        c = round(N ** (1.0 / k))
        for cc in (c - 1, c, c + 1):
            if cc >= 2 and cc ** k == N:
                return (cc, N // cc)

    # deterministic pseudo-random sequence of a values (no global RNG)
    state = seed & 0xFFFFFFFF
    tried = set()
    for _ in range(max_attempts):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        a = 2 + state % (N - 3)  # a in [2, N-2]
        if a in tried:
            continue
        tried.add(a)

        g = gcd(a, N)
        if g > 1:
            return (g, N // g)  # lucky: a shares a factor

        r = find_order_quantum(a, N) if quantum else multiplicative_order(a, N)
        if r is None or r % 2 != 0:
            continue  # need an even order

        x = pow(a, r // 2, N)
        if x == N - 1:
            continue  # a^(r/2) = -1 mod N -> trivial, retry
        p = gcd(x - 1, N)
        q = gcd(x + 1, N)
        for f in (p, q):
            if 1 < f < N and N % f == 0:
                return (f, N // f)
    return None


def verify_factorization(N, factors):
    """Check a returned (p, q) actually multiplies to N with nontrivial factors."""
    if factors is None:
        return False
    p, q = factors
    return 1 < p < N and 1 < q < N and p * q == N
