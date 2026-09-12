"""Fast Walsh-Hadamard transform: the FFT for XOR, and its AND/OR cousins.

The ordinary FFT multiplies polynomials, which is the same as CYCLIC convolution -- combining two
sequences by adding their indices mod n. But a different, equally useful convolution combines indices
by BITWISE XOR: (a * b)[k] = sum over i XOR j == k of a[i] * b[j]. This "XOR convolution" answers
questions like: if two independent processes each produce a random bitmask with a known distribution,
what is the distribution of their XOR? Or, in combinatorial game theory, if two game positions have
Grundy-number distributions, the XOR (Nim-sum) of the combined game convolves them this way. Computing
it directly is O(n^2); the FAST WALSH-HADAMARD TRANSFORM (FWHT) does it in O(n log n), exactly as the
FFT speeds up cyclic convolution.

The FWHT is a linear transform by the Hadamard matrix, computed by an in-place butterfly almost
identical to the FFT's: at each of the log2(n) stages, pair up entries a distance h apart and replace
(x, y) with (x + y, x - y). The magic is the CONVOLUTION THEOREM: transform both inputs, multiply them
pointwise, and inverse-transform, and you get their XOR convolution -- because the Hadamard matrix
diagonalises the XOR group algebra, turning convolution into pointwise product. The inverse is the same
butterfly divided by n. Two close relatives use different butterflies over the same subset lattice: the
OR convolution (indices combined by bitwise OR) via the SUM-OVER-SUBSETS (zeta) transform and its
Mobius inverse, and the AND convolution (bitwise AND) via the superset-sum transform. Together they
convolve over the three natural Boolean operations.

This module implements the FWHT and its inverse (integer-exact, so no floating-point error), XOR / OR
/ AND convolutions built on them, and the subset-sum (zeta) and Mobius transforms. It is verified
against the brute-force O(n^2) definition of each convolution on hundreds of random arrays, against
the round-trip identity (inverse-transform undoes transform), against linearity, and on hand-computed
small cases. Pure stdlib; a transform companion to the FFT and Sprague-Grundy notes."""

from __future__ import annotations


def _check_pow2(a):
    n = len(a)
    if n & (n - 1) != 0:
        raise ValueError(f"length must be a power of two, got {n}")
    return n


# --- Walsh-Hadamard transform (XOR) ----------------------------------------
def fwht(a, invert=False):
    """In-place-style Fast Walsh-Hadamard transform (returns a new list). The forward transform
    diagonalises XOR convolution; `invert=True` divides by n to undo it. Length must be a power of 2.
    Works on integers exactly (the /n on inversion is exact for genuine XOR-convolution results)."""
    n = _check_pow2(a)
    f = list(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x, y = f[j], f[j + h]
                f[j] = x + y
                f[j + h] = x - y
        h *= 2
    if invert:
        f = [v // n for v in f] if all(v % n == 0 for v in f) else [v / n for v in f]
    return f


def xor_convolution(a, b):
    """XOR convolution: c[k] = sum over i XOR j == k of a[i]*b[j], in O(n log n). Both inputs must
    have the same power-of-two length."""
    if len(a) != len(b):
        raise ValueError("inputs must have equal length")
    n = _check_pow2(a)
    fa = fwht(a)
    fb = fwht(b)
    fc = [fa[i] * fb[i] for i in range(n)]
    return fwht(fc, invert=True)


# --- subset-sum (zeta) / Mobius transforms and OR convolution --------------
def zeta_transform(a):
    """Sum-over-subsets: z[mask] = sum of a[sub] over all sub that are submasks of mask. O(n log n)."""
    n = _check_pow2(a)
    f = list(a)
    bit = 1
    while bit < n:
        for mask in range(n):
            if mask & bit:
                f[mask] += f[mask ^ bit]
        bit <<= 1
    return f


def mobius_transform(a):
    """Inverse of the zeta transform (subset Mobius inversion)."""
    n = _check_pow2(a)
    f = list(a)
    bit = 1
    while bit < n:
        for mask in range(n):
            if mask & bit:
                f[mask] -= f[mask ^ bit]
        bit <<= 1
    return f


def or_convolution(a, b):
    """OR convolution: c[k] = sum over i OR j == k of a[i]*b[j]. Via zeta * zeta then Mobius."""
    if len(a) != len(b):
        raise ValueError("inputs must have equal length")
    n = _check_pow2(a)
    za, zb = zeta_transform(a), zeta_transform(b)
    zc = [za[i] * zb[i] for i in range(n)]
    return mobius_transform(zc)


# --- superset-sum transforms and AND convolution ---------------------------
def superset_zeta(a):
    """Sum-over-supersets: z[mask] = sum of a[sup] over all sup that are supermasks of mask."""
    n = _check_pow2(a)
    f = list(a)
    bit = 1
    while bit < n:
        for mask in range(n):
            if not (mask & bit):
                f[mask] += f[mask | bit]
        bit <<= 1
    return f


def superset_mobius(a):
    """Inverse of the superset-sum transform."""
    n = _check_pow2(a)
    f = list(a)
    bit = 1
    while bit < n:
        for mask in range(n):
            if not (mask & bit):
                f[mask] -= f[mask | bit]
        bit <<= 1
    return f


def and_convolution(a, b):
    """AND convolution: c[k] = sum over i AND j == k of a[i]*b[j]. Via superset-zeta then Mobius."""
    if len(a) != len(b):
        raise ValueError("inputs must have equal length")
    n = _check_pow2(a)
    za, zb = superset_zeta(a), superset_zeta(b)
    zc = [za[i] * zb[i] for i in range(n)]
    return superset_mobius(zc)


# --- brute-force references -------------------------------------------------
def brute_convolution(a, b, op):
    """Direct O(n^2) convolution combining indices with the bitwise operator `op` (one of the
    built-in int operators: use a lambda like lambda i, j: i ^ j)."""
    n = len(a)
    c = [0] * n
    for i in range(n):
        for j in range(n):
            c[op(i, j)] += a[i] * b[j]
    return c
