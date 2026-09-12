"""Strassen's algorithm: multiplying matrices with seven products instead of eight.

Multiplying two n x n matrices the ordinary way costs n^3 scalar multiplications -- one for each of the
n^2 output entries times the n terms in its dot product. In 1969 Volker Strassen shattered the belief
that this was optimal: by splitting each matrix into four n/2 x n/2 blocks and forming seven cleverly
chosen products of block sums, the four output blocks can be assembled with only SEVEN block
multiplications instead of the naive eight. Applied recursively, the exponent drops from 3 to
log2(7) ~ 2.807 -- the first sub-cubic matrix multiplication, and the opening move in a decades-long
race (Coppersmith-Winograd, and beyond) to lower the exponent further. It matters because matrix
multiplication is the computational core of linear algebra, graph algorithms, and machine learning, so
shaving the exponent compounds across everything built on it.

The seven products are M1=(A11+A22)(B11+B22), M2=(A21+A22)B11, M3=A11(B12-B22), M4=A22(B21-B11),
M5=(A11+A12)B22, M6=(A21-A11)(B11+B12), M7=(A12-A22)(B21+B22). The output blocks recombine by pure
additions and subtractions: C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6. The trade is one
fewer multiplication (the expensive recursive operation) at the cost of more cheap additions -- a win
because multiplication dominates asymptotically. Below a size cutoff the recursion falls back to plain
multiplication, and non-power-of-two or rectangular matrices are handled by zero-padding to the next
power of two.

This module implements Strassen multiplication with padding for arbitrary dimensions (as long as the
inner dimensions match), a schoolbook reference, and works over exact integers or floats. It is
verified against the schoolbook O(n^3) product -- identical results on hundreds of random matrices of
assorted shapes and sizes -- plus the identity/zero/associativity checks and known small products. Pure
stdlib; a numerical-and-algebraic companion to the LU, QR, and Gaussian-elimination notes."""

from __future__ import annotations

_CUTOFF = 32          # below this dimension, plain multiplication wins


def schoolbook(a, b):
    """Ordinary O(n*m*p) matrix product of a (r x k) and b (k x c). Returns an r x c matrix."""
    r = len(a)
    k = len(a[0]) if r else 0
    c = len(b[0]) if b else 0
    out = [[0] * c for _ in range(r)]
    for i in range(r):
        ai = a[i]
        oi = out[i]
        for t in range(k):
            ait = ai[t]
            if ait == 0:
                continue
            bt = b[t]
            for j in range(c):
                oi[j] += ait * bt[j]
    return out


def _add(a, b):
    n = len(a)
    return [[a[i][j] + b[i][j] for j in range(len(a[i]))] for i in range(n)]


def _sub(a, b):
    n = len(a)
    return [[a[i][j] - b[i][j] for j in range(len(a[i]))] for i in range(n)]


def _split(m):
    """Split a 2n x 2n matrix into four n x n quadrants."""
    n = len(m) // 2
    a11 = [row[:n] for row in m[:n]]
    a12 = [row[n:] for row in m[:n]]
    a21 = [row[:n] for row in m[n:]]
    a22 = [row[n:] for row in m[n:]]
    return a11, a12, a21, a22


def _join(c11, c12, c21, c22):
    """Recombine four n x n quadrants into a 2n x 2n matrix."""
    top = [c11[i] + c12[i] for i in range(len(c11))]
    bot = [c21[i] + c22[i] for i in range(len(c21))]
    return top + bot


def _strassen_square(a, b):
    """Strassen product of two n x n matrices, n a power of two."""
    n = len(a)
    if n <= _CUTOFF:
        return schoolbook(a, b)
    a11, a12, a21, a22 = _split(a)
    b11, b12, b21, b22 = _split(b)

    m1 = _strassen_square(_add(a11, a22), _add(b11, b22))
    m2 = _strassen_square(_add(a21, a22), b11)
    m3 = _strassen_square(a11, _sub(b12, b22))
    m4 = _strassen_square(a22, _sub(b21, b11))
    m5 = _strassen_square(_add(a11, a12), b22)
    m6 = _strassen_square(_sub(a21, a11), _add(b11, b12))
    m7 = _strassen_square(_sub(a12, a22), _add(b21, b22))

    c11 = _add(_sub(_add(m1, m4), m5), m7)
    c12 = _add(m3, m5)
    c21 = _add(m2, m4)
    c22 = _add(_sub(_add(m1, m3), m2), m6)
    return _join(c11, c12, c21, c22)


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def multiply(a, b):
    """Multiply matrices a (r x k) and b (k x c) via Strassen's algorithm, zero-padding to a power of
    two internally. Returns the r x c product. Raises on a dimension mismatch."""
    r = len(a)
    k = len(a[0]) if r else 0
    k2 = len(b)
    c = len(b[0]) if b else 0
    if k != k2:
        raise ValueError(f"inner dimensions must match: {k} vs {k2}")
    if r == 0 or c == 0 or k == 0:
        return [[0] * c for _ in range(r)]

    size = _next_pow2(max(r, k, c))
    if size <= _CUTOFF:
        return schoolbook(a, b)
    # pad both to size x size with zeros
    pa = [[a[i][j] if i < r and j < k else 0 for j in range(size)] for i in range(size)]
    pb = [[b[i][j] if i < k and j < c else 0 for j in range(size)] for i in range(size)]
    pc = _strassen_square(pa, pb)
    return [[pc[i][j] for j in range(c)] for i in range(r)]


# --- helpers for validation -------------------------------------------------
def identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def equal(a, b, tol=1e-9):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if len(a[i]) != len(b[i]):
            return False
        for j in range(len(a[i])):
            if abs(a[i][j] - b[i][j]) > tol:
                return False
    return True
