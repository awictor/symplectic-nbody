"""Bluestein's algorithm -- the fast Fourier transform for ANY length, even a large prime.

The classic Cooley-Tukey FFT is fast only when the length factors nicely -- ideally a power of two. Feed
it a prime length like 1021 and it falls back to the O(n^2) naive sum, or forces you to zero-pad and
change the transform you are computing. Bluestein's algorithm (1968), also called the chirp z-transform,
removes that restriction: it computes the EXACT discrete Fourier transform of a sequence of ANY length n
in O(n log n) time, prime or not, by turning the DFT into a CONVOLUTION that a power-of-two FFT can do.

The trick is a piece of algebraic sleight of hand. The DFT exponent is n*k; Bluestein rewrites it with
the identity n*k = (n^2 + k^2 - (k-n)^2) / 2. Substituting splits the transform into three factors: a
"chirp" premultiply of the input by exp(-i pi n^2 / N), a convolution of that against a second chirp
sequence exp(+i pi m^2 / N), and a chirp postmultiply. The convolution is the only expensive part, and a
convolution of length n can be embedded in a circular convolution of any length >= 2n-1 -- so we pick the
next power of two, run the fast radix-2 FFT there, multiply pointwise, and invert. The chirps carry all
the length-n structure; the FFT engine underneath only ever sees a friendly power-of-two length.

Why "chirp"? The sequence exp(i pi m^2 / N) is a discretely sampled linear frequency sweep -- a chirp, as
in radar. The chirp z-transform generalises further: it evaluates the z-transform along any spiral or
arc of points in the complex plane, not just the equally spaced points on the unit circle that the DFT
uses, which is why it underlies zoom-FFT spectral analysis. This module implements the plain
arbitrary-length DFT and its inverse via Bluestein, plus a linear convolution of two arbitrary-length
sequences built on the same machinery.

Everything rests on a small self-contained radix-2 FFT (iterative, bit-reversal permutation) so the
module is standalone; it does not import the repository's other FFT. Complex arithmetic uses Python's
built-in ``complex`` and ``cmath``.

Validation. (1) Against a direct O(n^2) DFT computed straight from the definition, for many lengths --
powers of two, odd composites, and PRIMES (7, 13, 101, 251) -- the Bluestein output matches to machine
precision, which is the whole point: correct transforms at lengths the radix-2 FFT cannot handle
directly. (2) The round trip idft(dft(x)) == x for random complex and real inputs at arbitrary lengths.
(3) Linearity and the known transform of simple signals (a constant maps to a spike at bin 0; a pure
complex exponential of integer frequency maps to a single nonzero bin). (4) Bluestein convolution
matches a naive O(n*m) convolution for arbitrary length pairs. Pure standard library."""

import cmath
import math


# ---------------------------------------------------------------------------
# a small self-contained radix-2 FFT (iterative, in-place)
# ---------------------------------------------------------------------------

def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _fft_pow2(a, inverse=False):
    """In-place iterative radix-2 FFT of a list whose length is a power of two. Returns a new list."""
    n = len(a)
    if n == 1:
        return [a[0]]
    if n & (n - 1) != 0:
        raise ValueError("length must be a power of two")

    a = list(a)
    # bit-reversal permutation
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    sign = 1 if inverse else -1
    while length <= n:
        ang = sign * 2 * math.pi / length
        wlen = cmath.exp(1j * ang)
        for start in range(0, n, length):
            w = 1 + 0j
            half = length >> 1
            for k in range(half):
                u = a[start + k]
                v = a[start + k + half] * w
                a[start + k] = u + v
                a[start + k + half] = u - v
                w *= wlen
        length <<= 1

    if inverse:
        a = [x / n for x in a]
    return a


# ---------------------------------------------------------------------------
# Bluestein DFT for arbitrary length
# ---------------------------------------------------------------------------

def dft(x):
    """Discrete Fourier transform of a sequence of ANY length via Bluestein's algorithm. O(n log n).

    Accepts real or complex input; returns a list of complex values X[k] = sum_j x[j] exp(-2πi jk/n).
    """
    n = len(x)
    if n == 0:
        return []
    if n == 1:
        return [complex(x[0])]

    # chirp: w[j] = exp(-i pi j^2 / n)   (note pi, not 2pi, from the (n^2+k^2-(k-n)^2)/2 split)
    a = [complex(x[j]) * cmath.exp(-1j * math.pi * (j * j % (2 * n)) / n) for j in range(n)]

    m = _next_pow2(2 * n - 1)

    # b[j] = exp(+i pi j^2 / n) as a symmetric filter for circular convolution
    b = [0j] * m
    b[0] = cmath.exp(1j * math.pi * 0 / n)
    for j in range(1, n):
        val = cmath.exp(1j * math.pi * (j * j % (2 * n)) / n)
        b[j] = val
        b[m - j] = val   # wrap for the negative indices in circular convolution

    # linear convolution a * b via FFT, then pick the length-n aligned window
    fa = _fft_pow2(a + [0j] * (m - n))
    fb = _fft_pow2(b)
    fc = [fa[i] * fb[i] for i in range(m)]
    c = _fft_pow2(fc, inverse=True)

    # postmultiply by the chirp again
    X = []
    for k in range(n):
        chirp = cmath.exp(-1j * math.pi * (k * k % (2 * n)) / n)
        X.append(chirp * c[k])
    return X


def idft(X):
    """Inverse DFT of an arbitrary-length sequence. Returns complex values; x[j] = (1/n) sum_k X[k] e^{+2πi jk/n}."""
    n = len(X)
    if n == 0:
        return []
    # conjugate trick: idft(X) = conj(dft(conj(X))) / n
    conj = [z.conjugate() for z in X]
    y = dft(conj)
    return [z.conjugate() / n for z in y]


# ---------------------------------------------------------------------------
# direct DFT reference (O(n^2)) -- used to validate Bluestein
# ---------------------------------------------------------------------------

def dft_direct(x):
    """Straight-from-the-definition O(n^2) DFT, for validation."""
    n = len(x)
    out = []
    for k in range(n):
        s = 0j
        for j in range(n):
            s += complex(x[j]) * cmath.exp(-2j * math.pi * j * k / n)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# linear convolution of arbitrary-length sequences
# ---------------------------------------------------------------------------

def convolve(a, b):
    """Linear convolution of two arbitrary-length sequences via Bluestein DFTs. Returns len(a)+len(b)-1
    complex values (real if inputs are real, up to rounding)."""
    if not a or not b:
        return []
    n = len(a) + len(b) - 1
    fa = dft(list(a) + [0j] * (n - len(a)))
    fb = dft(list(b) + [0j] * (n - len(b)))
    fc = [fa[i] * fb[i] for i in range(n)]
    return idft(fc)


def convolve_direct(a, b):
    """Naive O(n*m) convolution, for validation."""
    n = len(a) + len(b) - 1
    out = [0j] * n
    for i in range(len(a)):
        for j in range(len(b)):
            out[i + j] += complex(a[i]) * complex(b[j])
    return out
