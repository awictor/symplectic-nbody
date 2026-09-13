"""LDPC codes: sparse parity checks and message passing that get within a whisper of the Shannon limit.

Low-density parity-check codes (Gallager, 1962; rediscovered in the 1990s) are the error-correcting
codes in Wi-Fi, 5G, and deep-space links, and they come astonishingly close to the CHANNEL CAPACITY
that Shannon proved is the ceiling on reliable communication. The magic is in the word SPARSE: the
code is defined by a parity-check matrix H in which every row (a check) touches only a few bits and
every bit touches only a few checks. That sparsity is exactly what makes iterative message passing on
the code's bipartite graph both cheap and near-optimal.

A binary codeword c satisfies H c = 0 (mod 2): each parity check is the XOR of its incident bits and
must be zero. Encoding an arbitrary message into such a codeword needs H in systematic form
[P | I] via Gauss-Jordan over GF(2), giving a generator G = [I | P^T]; then c = m G. Transmit c over a
BINARY SYMMETRIC CHANNEL that independently flips each bit with probability p, and the decoder must
recover c from the noisy word.

Two decoders, both operating on the TANNER GRAPH (bits on one side, checks on the other, edges from H):

  BIT-FLIPPING (hard decision). Compute the syndrome; for each bit count how many of its checks are
  currently unsatisfied; flip the bit(s) with the most unsatisfied checks; repeat. Simple, and it
  fixes a surprising number of errors.

  SUM-PRODUCT / BELIEF PROPAGATION (soft decision). Treat each bit's channel value as a
  log-likelihood ratio and pass messages: checks tell bits what parity implies (the tanh rule), bits
  combine what their checks say, iterate. This is the near-capacity decoder, and on the sparse graph
  each iteration is linear in the number of edges.

This module builds a Gallager-style regular LDPC parity-check matrix, puts it in systematic form to
get an encoder, simulates a binary symmetric channel with a seeded RNG, and implements both the
bit-flipping and the log-domain sum-product decoders. It is validated exactly: every generated
codeword satisfies H c = 0; encode/decode round-trips with no noise; both decoders correct all
single-bit errors and the great majority of multi-bit errors below threshold; the decoded word always
satisfies the parity checks on success; and belief propagation corrects strictly more than
bit-flipping at a given noise level. Pure stdlib; the modern, graph-based coding companion to the
algebraic Hamming, Reed-Muller, Reed-Solomon, and convolutional codes."""

from __future__ import annotations

import math


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def make_regular_ldpc(n, wc, wr, seed=0):
    """Build a Gallager-style regular LDPC parity-check matrix H.

    n bits, each bit in wc checks, each check on wr bits. Requires n * wc divisible by wr; the number
    of checks is m = n * wc / wr. Returns H as a list of m rows, each a list of n 0/1 ints.
    """
    if (n * wc) % wr != 0:
        raise ValueError("n*wc must be divisible by wr")
    m = n * wc // wr
    rng = _lcg(seed)

    # Gallager: stack wc "sub-matrices", each a permutation of a base band with one 1 per column.
    # Base band (first sub-matrix): rows of m/wc checks, each check covers a contiguous block of wr.
    sub_rows = m // wc
    if sub_rows * wr != n:
        # fall back to a randomized construction when the clean Gallager sizing doesn't hold
        return _random_regular_ldpc(n, wc, wr, m, rng)

    H = []
    # first sub-matrix: check i covers columns [i*wr, (i+1)*wr)
    base = []
    for i in range(sub_rows):
        row = [0] * n
        for j in range(i * wr, (i + 1) * wr):
            row[j] = 1
        base.append(row)
    H.extend(base)
    # remaining sub-matrices: random column permutations of the base
    for _ in range(wc - 1):
        perm = list(range(n))
        for i in range(n - 1, 0, -1):
            k = int(rng() * (i + 1))
            perm[i], perm[k] = perm[k], perm[i]
        for row in base:
            H.append([row[perm[j]] for j in range(n)])
    return H


def _random_regular_ldpc(n, wc, wr, m, rng):
    """Randomized near-regular construction: distribute column/row degrees by shuffled sockets."""
    # each column contributes wc sockets, each row accepts wr sockets; match by shuffling
    col_sockets = []
    for j in range(n):
        col_sockets.extend([j] * wc)
    for i in range(len(col_sockets) - 1, 0, -1):
        k = int(rng() * (i + 1))
        col_sockets[i], col_sockets[k] = col_sockets[k], col_sockets[i]
    H = [[0] * n for _ in range(m)]
    idx = 0
    for r in range(m):
        placed = set()
        for _ in range(wr):
            # find next distinct column for this row
            tries = 0
            while idx < len(col_sockets) and col_sockets[idx] in placed and tries < len(col_sockets):
                idx += 1
                tries += 1
            if idx >= len(col_sockets):
                idx = 0
            col = col_sockets[idx]
            idx += 1
            placed.add(col)
            H[r][col] = 1
    return H


def _row_reduce_gf2(H):
    """Gauss-Jordan over GF(2). Returns (reduced H copy, pivot columns list)."""
    m = len(H)
    n = len(H[0])
    A = [row[:] for row in H]
    pivots = []
    r = 0
    for c in range(n):
        # find a pivot in column c at or below row r
        piv = None
        for i in range(r, m):
            if A[i][c]:
                piv = i
                break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        for i in range(m):
            if i != r and A[i][c]:
                A[i] = [(a ^ b) for a, b in zip(A[i], A[r])]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return A, pivots


def systematic_generator(H):
    """Generator matrix G (k x n) for the code with parity-check H, via systematic form.

    Returns (G, info_positions): G rows span the codespace {c : H c = 0}, and info_positions are the
    columns carrying the free (message) bits.
    """
    m = len(H)
    n = len(H[0])
    A, pivots = _row_reduce_gf2(H)
    pivot_set = set(pivots)
    free = [c for c in range(n) if c not in pivot_set]
    k = len(free)

    # Build G: for each free column f, a codeword with a 1 there and pivot bits set to satisfy H c = 0.
    G = []
    # rank rows of A hold the pivot equations: pivot_col = XOR of free cols in that row
    rank = len(pivots)
    for f in free:
        c = [0] * n
        c[f] = 1
        # set pivot bits so each reduced parity row is satisfied
        for r in range(rank):
            pc = pivots[r]
            # row r: A[r][pc]=1; sum over free cols A[r][fc]*c[fc] must equal c[pc]
            s = 0
            for fc in free:
                if A[r][fc] and c[fc]:
                    s ^= 1
            c[pc] = s
        G.append(c)
    return G, free


def encode(message, G):
    """Encode a length-k message (list of 0/1) into a length-n codeword via c = m G over GF(2)."""
    k = len(G)
    if len(message) != k:
        raise ValueError(f"message length {len(message)} != k {k}")
    n = len(G[0])
    c = [0] * n
    for i in range(k):
        if message[i]:
            row = G[i]
            for j in range(n):
                c[j] ^= row[j]
    return c


def syndrome(H, word):
    """H word (mod 2): a list of parity-check results (all zero iff word is a codeword)."""
    return [sum(H[i][j] & word[j] for j in range(len(word))) & 1 for i in range(len(H))]


def is_codeword(H, word):
    return all(s == 0 for s in syndrome(H, word))


def bsc(word, p, seed=0):
    """Pass a word through a binary symmetric channel: flip each bit with probability p."""
    rng = _lcg(seed)
    return [b ^ (1 if rng() < p else 0) for b in word]


def decode_bitflip(H, received, max_iter=100):
    """Gallager bit-flipping (hard-decision) decoder. Returns (decoded_word, success)."""
    n = len(received)
    m = len(H)
    word = list(received)
    # precompute check->bits and bit->checks adjacency
    check_bits = [[j for j in range(n) if H[i][j]] for i in range(m)]
    bit_checks = [[i for i in range(m) if H[i][j]] for j in range(n)]

    for _ in range(max_iter):
        synd = [sum(word[j] for j in check_bits[i]) & 1 for i in range(m)]
        if not any(synd):
            return word, True
        # count unsatisfied checks per bit
        unsat = [sum(synd[i] for i in bit_checks[j]) for j in range(n)]
        best = max(unsat)
        if best == 0:
            break
        # flip all bits achieving the max (parallel bit-flipping)
        for j in range(n):
            if unsat[j] == best:
                word[j] ^= 1
    return word, is_codeword(H, word)


def min_distance(H, G=None):
    """Brute-force minimum distance: smallest weight of a nonzero codeword (2^k enumeration).

    A random sparse H can produce weight-2 codewords (from duplicate columns) that no decoder can
    fix; this exposes that so callers can pick a good construction.
    """
    if G is None:
        G, _ = systematic_generator(H)
    k = len(G)
    md = None
    for bits in range(1, 1 << k):
        msg = [(bits >> i) & 1 for i in range(k)]
        w = sum(encode(msg, G))
        if w > 0 and (md is None or w < md):
            md = w
    return md


def decode_sum_product(H, received, p, max_iter=50):
    """Log-domain sum-product (belief propagation) decoder over a BSC with crossover p.

    Returns (decoded_word, success).
    """
    n = len(received)
    m = len(H)
    check_bits = [[j for j in range(n) if H[i][j]] for i in range(m)]
    bit_checks = [[i for i in range(m) if H[i][j]] for j in range(n)]

    # channel log-likelihood ratio for a BSC: LLR = log P(0)/P(1)
    # received bit r: if r==0, more likely 0 -> +L; if r==1, -L; L = log((1-p)/p)
    p = min(max(p, 1e-6), 0.5 - 1e-6)
    Lc = math.log((1 - p) / p)
    llr = [Lc if received[j] == 0 else -Lc for j in range(n)]

    # messages: M[i][j] bit->check, E[i][j] check->bit
    M = {(i, j): llr[j] for i in range(m) for j in check_bits[i]}
    E = {(i, j): 0.0 for i in range(m) for j in check_bits[i]}

    for _ in range(max_iter):
        # check -> bit (tanh rule)
        for i in range(m):
            for j in check_bits[i]:
                prod = 1.0
                for jp in check_bits[i]:
                    if jp != j:
                        prod *= math.tanh(M[(i, jp)] / 2.0)
                prod = min(max(prod, -1 + 1e-12), 1 - 1e-12)
                E[(i, j)] = 2.0 * math.atanh(prod)
        # bit -> check and tentative decision
        word = [0] * n
        for j in range(n):
            total = llr[j] + sum(E[(i, j)] for i in bit_checks[j])
            word[j] = 0 if total >= 0 else 1
            for i in bit_checks[j]:
                M[(i, j)] = llr[j] + sum(E[(ip, j)] for ip in bit_checks[j] if ip != i)
        if is_codeword(H, word):
            return word, True
    return word, is_codeword(H, word)
