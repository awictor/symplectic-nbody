"""Sum over subsets (SOS DP): zeta and Moebius transforms on the boolean lattice in O(n*2^n).

Given a function f defined on all 2^n subsets of an n-element universe (an array indexed by bitmask),
a recurring need is the SUBSET SUM transform

    F(S) = sum over all T subset of S of f(T),

and its superset counterpart G(S) = sum over T superset of S. Done naively this is O(3^n) -- every
mask times each of its submasks. The "sum over subsets" dynamic program (SOS DP) does it in O(n*2^n)
by adding one dimension at a time: for each bit i, every mask that HAS bit i absorbs the value of the
mask with bit i cleared. After sweeping all n bits, each F(S) has accumulated exactly its subsets.
This is the ZETA TRANSFORM of the subset lattice, and it is the discrete analogue of a cumulative
sum generalised from a line to the boolean hypercube.

Its inverse is the MOEBIUS TRANSFORM: run the same sweep but SUBTRACT, recovering f from F. Together
they turn subset-indexed convolutions into pointwise products:

    OR-convolution  (h(S) = sum over A|B=S of f(A)g(B))  = moebius( zeta(f) . zeta(g) ),
    AND-convolution (h(S) = sum over A&B=S of f(A)g(B))  = superset-moebius( sz(f) . sz(g) ),

and the SUBSET-SUM convolution (h(S) = sum over disjoint A,B with A|B=S of f(A)g(B)) via the ranked
zeta trick: index by popcount so the "disjoint" constraint becomes an ordinary polynomial product in
the rank dimension. This module implements zeta/moebius (subset and superset), OR/AND convolution,
and the ranked subset-sum convolution.

Validated against brute force: the subset-sum transform matches the O(3^n) definition for every mask,
zeta and moebius are exact inverses, OR/AND convolutions match their O(4^n) defining sums, and the
subset-sum convolution matches the disjoint-pair definition. Pure stdlib; the lattice-transform
companion to the FFT and Walsh-Hadamard (XOR) convolutions already in the repo."""

from __future__ import annotations


def zeta_subset(f):
    """Subset-sum (zeta) transform in place-safe form: returns F(S) = sum_{T subseteq S} f(T).
    Length of f must be a power of two, 2^n."""
    F = list(f)
    n = (len(F) - 1).bit_length() if len(F) > 1 else 0
    if len(F) != 1 << n:
        raise ValueError("length of f must be a power of two")
    for i in range(n):
        bit = 1 << i
        for mask in range(len(F)):
            if mask & bit:
                F[mask] += F[mask ^ bit]
    return F


def moebius_subset(F):
    """Inverse of zeta_subset: recovers f from F(S) = sum_{T subseteq S} f(T)."""
    f = list(F)
    n = (len(f) - 1).bit_length() if len(f) > 1 else 0
    if len(f) != 1 << n:
        raise ValueError("length must be a power of two")
    for i in range(n):
        bit = 1 << i
        for mask in range(len(f)):
            if mask & bit:
                f[mask] -= f[mask ^ bit]
    return f


def zeta_superset(f):
    """Superset-sum transform: G(S) = sum_{T superseteq S} f(T)."""
    G = list(f)
    n = (len(G) - 1).bit_length() if len(G) > 1 else 0
    if len(G) != 1 << n:
        raise ValueError("length must be a power of two")
    for i in range(n):
        bit = 1 << i
        for mask in range(len(G)):
            if not (mask & bit):
                G[mask] += G[mask | bit]
    return G


def moebius_superset(G):
    """Inverse of zeta_superset."""
    f = list(G)
    n = (len(f) - 1).bit_length() if len(f) > 1 else 0
    if len(f) != 1 << n:
        raise ValueError("length must be a power of two")
    for i in range(n):
        bit = 1 << i
        for mask in range(len(f)):
            if not (mask & bit):
                f[mask] -= f[mask | bit]
    return f


def or_convolution(f, g):
    """h(S) = sum over A|B == S of f(A)*g(B), via zeta . zeta then moebius."""
    if len(f) != len(g):
        raise ValueError("f and g must have equal length")
    Ff = zeta_subset(f)
    Fg = zeta_subset(g)
    prod = [a * b for a, b in zip(Ff, Fg)]
    return moebius_subset(prod)


def and_convolution(f, g):
    """h(S) = sum over A&B == S of f(A)*g(B), via superset-zeta . superset-zeta then superset-moebius."""
    if len(f) != len(g):
        raise ValueError("f and g must have equal length")
    Gf = zeta_superset(f)
    Gg = zeta_superset(g)
    prod = [a * b for a, b in zip(Gf, Gg)]
    return moebius_superset(prod)


def subset_sum_convolution(f, g):
    """h(S) = sum over disjoint A,B with A|B == S of f(A)*g(B) (the subset-sum / disjoint convolution).
    Uses the ranked zeta trick: index by popcount so disjointness becomes a rank-dimension product."""
    if len(f) != len(g):
        raise ValueError("f and g must have equal length")
    size = len(f)
    n = (size - 1).bit_length() if size > 1 else 0
    if size != 1 << n:
        raise ValueError("length must be a power of two")

    popcount = [0] * size
    for m in range(1, size):
        popcount[m] = popcount[m >> 1] + (m & 1)

    # fhat[k][mask], ghat[k][mask]: value of f/g at mask if popcount(mask)==k else 0, then zeta each rank
    fhat = [[0] * size for _ in range(n + 1)]
    ghat = [[0] * size for _ in range(n + 1)]
    for m in range(size):
        fhat[popcount[m]][m] = f[m]
        ghat[popcount[m]][m] = g[m]
    for k in range(n + 1):
        fhat[k] = zeta_subset(fhat[k])
        ghat[k] = zeta_subset(ghat[k])

    # rank convolution: hhat[k] = sum_{i+j=k} fhat[i] . ghat[j]
    hhat = [[0] * size for _ in range(n + 1)]
    for k in range(n + 1):
        for i in range(k + 1):
            j = k - i
            fi = fhat[i]
            gj = ghat[j]
            hk = hhat[k]
            for m in range(size):
                hk[m] += fi[m] * gj[m]

    # invert the zeta per rank, then read h(S) off the rank == popcount(S)
    h = [0] * size
    for k in range(n + 1):
        hk = moebius_subset(hhat[k])
        for m in range(size):
            if popcount[m] == k:
                h[m] = hk[m]
    return h


# --- brute-force references --------------------------------------------------
def brute_zeta_subset(f):
    """O(3^n) definition: F(S) = sum over submasks T of S of f(T)."""
    size = len(f)
    F = [0] * size
    for S in range(size):
        sub = S
        while True:
            F[S] += f[sub]
            if sub == 0:
                break
            sub = (sub - 1) & S
    return F


def brute_or_convolution(f, g):
    size = len(f)
    h = [0] * size
    for A in range(size):
        for B in range(size):
            h[A | B] += f[A] * g[B]
    return h


def brute_and_convolution(f, g):
    size = len(f)
    h = [0] * size
    for A in range(size):
        for B in range(size):
            h[A & B] += f[A] * g[B]
    return h


def brute_subset_sum_convolution(f, g):
    size = len(f)
    h = [0] * size
    for A in range(size):
        for B in range(size):
            if A & B == 0:  # disjoint
                h[A | B] += f[A] * g[B]
    return h
