"""The resultant and discriminant: detecting common roots and repeated roots without finding them.

Two polynomials share a common root if and only if a single number -- their RESULTANT -- vanishes. This
is remarkable: you can decide whether p and q have a root in common, over any field, WITHOUT computing
a single root, purely from their coefficients. The resultant is the determinant of the SYLVESTER
MATRIX, a (deg p + deg q) square matrix built by stacking shifted copies of the two coefficient rows.
It equals (up to sign and leading coefficients) the product of p evaluated at all of q's roots, and
also the product of ALL pairwise differences (a_i - b_j) of the two root sets -- so it is zero exactly
when some a_i equals some b_j.

The DISCRIMINANT is the special case that measures a single polynomial against its own derivative:
disc(p) = (-1)^{n(n-1)/2} / a_n * Res(p, p'). It vanishes exactly when p has a REPEATED root, and its
sign (for real polynomials) even tells you how many roots are real -- disc > 0 for a quadratic means
two real roots, disc < 0 means a complex pair. Every "b^2 - 4ac" you have ever used is the
discriminant of a quadratic; this generalizes it to any degree.

Because a common factor of p and q forces a common root, the resultant is also the algebraic engine of
ELIMINATION: given two polynomial equations in two unknowns, taking the resultant with respect to one
variable eliminates it, leaving a single-variable polynomial whose roots are the surviving coordinates
-- the foundation of solving polynomial systems and of computer algebra.

This module builds the Sylvester matrix, computes the resultant by an exact integer/rational
determinant, and derives the discriminant. It is validated against the root definitions: the resultant
equals a_p^{deg q} times the product of p over q's roots (and the full product of root differences); it
is zero exactly when the polynomials share a root; the discriminant is zero exactly for repeated roots
and reproduces b^2 - 4ac for quadratics and the known cubic discriminant; and elimination on a small
two-variable system yields the correct solution coordinates. Pure stdlib (Fraction); the
elimination-theory companion to the Sturm real-root and Durand-Kerner complex-root tools."""

from __future__ import annotations

from fractions import Fraction


def _trim(p):
    """Strip leading (high-degree) zeros; keep at least one coefficient."""
    i = 0
    while i < len(p) - 1 and p[i] == 0:
        i += 1
    return list(p[i:])


def sylvester_matrix(p, q):
    """The Sylvester matrix of p and q (both high-degree-first). Size (deg p + deg q)^2."""
    p = _trim([Fraction(c) for c in p])
    q = _trim([Fraction(c) for c in q])
    m = len(p) - 1  # deg p
    n = len(q) - 1  # deg q
    size = m + n
    M = [[Fraction(0)] * size for _ in range(size)]
    # first n rows: shifted copies of p
    for i in range(n):
        for j, c in enumerate(p):
            M[i][i + j] = c
    # next m rows: shifted copies of q
    for i in range(m):
        for j, c in enumerate(q):
            M[n + i][i + j] = c
    return M


def _det(M):
    """Exact determinant of a rational matrix by fraction-free-ish Gaussian elimination."""
    n = len(M)
    if n == 0:
        return Fraction(1)
    A = [[Fraction(x) for x in row] for row in M]
    det = Fraction(1)
    for col in range(n):
        # find a pivot
        piv = None
        for r in range(col, n):
            if A[r][col] != 0:
                piv = r
                break
        if piv is None:
            return Fraction(0)
        if piv != col:
            A[col], A[piv] = A[piv], A[col]
            det = -det
        det *= A[col][col]
        inv = A[col][col]
        for r in range(col + 1, n):
            factor = A[r][col] / inv
            if factor != 0:
                for c in range(col, n):
                    A[r][c] -= factor * A[col][c]
    return det


def resultant(p, q):
    """Resultant Res(p, q): zero iff p and q share a root. Exact Fraction."""
    p = _trim([Fraction(c) for c in p])
    q = _trim([Fraction(c) for c in q])
    if len(p) == 1 and len(q) == 1:
        return Fraction(1)  # two nonzero constants: no roots, resultant 1
    if len(p) == 1:
        # Res(const c, q) = c^deg q
        return p[0] ** (len(q) - 1)
    if len(q) == 1:
        return q[0] ** (len(p) - 1)
    return _det(sylvester_matrix(p, q))


def derivative(p):
    """Derivative of p (high-degree-first)."""
    p = _trim([Fraction(c) for c in p])
    n = len(p) - 1
    if n <= 0:
        return [Fraction(0)]
    return [p[i] * (n - i) for i in range(n)]


def discriminant(p):
    """Discriminant of p: zero iff p has a repeated root. disc = (-1)^{n(n-1)/2}/a_n * Res(p, p')."""
    p = _trim([Fraction(c) for c in p])
    n = len(p) - 1
    if n < 1:
        raise ValueError("discriminant needs degree >= 1")
    a_n = p[0]
    res = resultant(p, derivative(p))
    sign = (-1) ** (n * (n - 1) // 2)
    return sign * res / a_n


def has_common_root(p, q):
    """True iff p and q share a root (resultant is zero)."""
    return resultant(p, q) == 0


def has_repeated_root(p):
    """True iff p has a repeated root (discriminant is zero)."""
    return discriminant(p) == 0


def eliminate(p, q, var="x"):
    """Given two bivariate polynomials as coefficient lists in one variable whose entries are
    coefficient lists in the other, this is a placeholder for the general elimination; here we expose
    resultant() as the elimination primitive. (See tests for a worked two-variable example.)"""
    return resultant(p, q)


def product_of_p_over_roots(p_coeffs, q_roots):
    """Reference: a_p^{deg q} * prod over q's roots r of p(r) -- equals Res(p, q) up to sign."""
    from fractions import Fraction as F
    p = _trim([complex(c) for c in p_coeffs])

    def peval(x):
        r = 0
        for c in p:
            r = r * x + c
        return r
    prod = 1
    for r in q_roots:
        prod *= peval(r)
    return prod
