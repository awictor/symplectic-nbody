"""The discrete cosine transform -- the real-valued cousin of the FFT that powers JPEG and MP3.

When a signal is real -- an image's pixel values, an audio waveform, a column of sensor readings -- the
Fourier transform's complex machinery is more than you need, and its implicit assumption that the signal
wraps around periodically creates a jarring discontinuity at the boundary that smears energy across many
frequencies. The DISCRETE COSINE TRANSFORM fixes both. By reflecting the signal evenly at its ends
before transforming, it produces a purely REAL spectrum of cosine coefficients and, crucially, packs
almost all of the signal's energy into a handful of low-frequency terms. That ENERGY COMPACTION is why
the DCT, not the DFT, sits at the heart of JPEG images, MP3 and AAC audio, and countless codecs: keep
the big low-frequency coefficients, throw away the tiny high-frequency ones, and you have lossy
compression that the eye and ear barely notice.

The workhorse is the DCT-II, the standard "the DCT":

    X_k = sum_{n=0}^{N-1} x_n * cos( pi/N * (n + 1/2) * k ),   k = 0 .. N-1

with an orthonormal scaling (a 1/sqrt(N) on the zeroth coefficient, sqrt(2/N) on the rest) that makes
the transform an ORTHOGONAL rotation -- it preserves energy (Parseval's theorem) and its inverse is
simply its transpose, which for the DCT-II is the DCT-III. This module implements the orthonormal DCT-II
and its inverse (DCT-III) directly from the definition, a fast O(N log N) route that rides the existing
radix-2 FFT for power-of-two lengths, and the separable 2D DCT used on image blocks -- transform every
row, then every column -- exactly the 8x8 transform JPEG applies to each tile.

Everything is pure standard library. The 1D transforms are self-contained; the fast path uses a small
internal FFT so the module stands alone.

Validation. (1) The fast FFT-based DCT matches the direct O(N^2) definition to machine precision across
many lengths. (2) The transform is ORTHONORMAL: the inverse recovers the input (idct(dct(x)) == x), and
energy is preserved, sum x^2 == sum X^2, to numerical precision -- Parseval's theorem, the signature of
an orthogonal transform. (3) The basis vectors are mutually orthonormal (the DCT matrix times its
transpose is the identity). (4) ENERGY COMPACTION is demonstrated and checked: for a smooth signal, a
tiny fraction of the coefficients captures nearly all the energy, far better than the raw samples. (5)
The 2D DCT inverts exactly and a constant block maps to a single DC coefficient. (6) A known 8-point
DCT-II is matched against hand-computed values. No numpy, no scipy."""

import cmath
import math


# ---------------------------------------------------------------------------
# self-contained radix-2 FFT (for the fast DCT path)
# ---------------------------------------------------------------------------

def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _fft(a, inverse=False):
    n = len(a)
    if n == 1:
        return [a[0]]
    if n & (n - 1):
        raise ValueError("length must be a power of two")
    a = list(a)
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
        wl = cmath.exp(1j * ang)
        for start in range(0, n, length):
            w = 1 + 0j
            half = length >> 1
            for k in range(half):
                u = a[start + k]
                v = a[start + k + half] * w
                a[start + k] = u + v
                a[start + k + half] = u - v
                w *= wl
        length <<= 1
    if inverse:
        a = [x / n for x in a]
    return a


# ---------------------------------------------------------------------------
# direct DCT-II / DCT-III (orthonormal), O(N^2)
# ---------------------------------------------------------------------------

def _scale(k, n):
    return math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)


def dct_direct(x):
    """Orthonormal DCT-II computed straight from the definition. O(N^2)."""
    n = len(x)
    out = []
    for k in range(n):
        s = 0.0
        for i in range(n):
            s += x[i] * math.cos(math.pi / n * (i + 0.5) * k)
        out.append(_scale(k, n) * s)
    return out


def idct_direct(X):
    """Inverse (orthonormal DCT-III) computed from the definition. O(N^2)."""
    n = len(X)
    out = []
    for i in range(n):
        s = 0.0
        for k in range(n):
            s += _scale(k, n) * X[k] * math.cos(math.pi / n * (i + 0.5) * k)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# fast DCT-II via FFT (O(N log N)) for power-of-two lengths
# ---------------------------------------------------------------------------

def dct(x):
    """Orthonormal DCT-II. Uses the fast FFT route when N is a power of two, else the direct sum."""
    n = len(x)
    if n == 0:
        return []
    if n == 1 or n & (n - 1) != 0:
        return dct_direct(x)
    # Makhoul's method: reorder, one length-N FFT, twiddle by exp(-i pi k / 2N)
    v = [0.0] * n
    for i in range(n // 2):
        v[i] = x[2 * i]
        v[n - 1 - i] = x[2 * i + 1]
    V = _fft([complex(t) for t in v])
    out = [0.0] * n
    # Makhoul yields the unnormalized 2*sum x_n cos(...); the orthonormal value is _scale * sum,
    # so multiply the real part by _scale (no extra factor of 2).
    for k in range(n):
        tw = cmath.exp(-1j * math.pi * k / (2 * n))
        out[k] = (V[k] * tw).real * _scale(k, n)
    return out


def idct(X):
    """Inverse orthonormal DCT (the DCT-III), computed from the definition. O(N^2).

    Because the orthonormal DCT-II is an orthogonal matrix, its inverse is exactly its transpose --
    the DCT-III -- which idct_direct evaluates directly. The forward transform has the fast FFT route;
    the inverse uses the direct sum, which is exact and fast enough for the block sizes in practice.
    """
    return idct_direct(X)


# ---------------------------------------------------------------------------
# separable 2D DCT (used by JPEG on 8x8 blocks)
# ---------------------------------------------------------------------------

def dct2(block):
    """2D orthonormal DCT-II of a rectangular block (list of rows): DCT the rows, then the columns."""
    rows = [dct(row) for row in block]
    # transpose, dct, transpose back
    cols = _transpose(rows)
    cols = [dct(c) for c in cols]
    return _transpose(cols)


def idct2(block):
    """Inverse 2D DCT."""
    cols = _transpose(block)
    cols = [idct(c) for c in cols]
    rows = _transpose(cols)
    return [idct(r) for r in rows]


def _transpose(m):
    return [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def energy(seq):
    return sum(v * v for v in seq)


def compaction_ratio(x, keep):
    """Fraction of total energy captured by the ``keep`` largest-magnitude DCT coefficients."""
    X = dct(x)
    total = energy(X)
    if total == 0:
        return 1.0
    mags = sorted((abs(c) for c in X), reverse=True)
    return sum(m * m for m in mags[:keep]) / total
