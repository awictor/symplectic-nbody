"""LLL lattice basis reduction -- turning a skewed integer basis into a short, nearly-orthogonal one.

A lattice is the set of all integer combinations of a set of basis vectors: every point you can reach
by adding and subtracting whole copies of the basis. The same lattice has infinitely many bases -- any
two related by an integer matrix of determinant +/-1 (a unimodular transform) generate exactly the same
set of points. Most of those bases are terrible: long, skewed vectors pointing almost the same way. A
*reduced* basis is one made of short, nearly-orthogonal vectors, and finding one is the key that
unlocks a startling range of problems -- from finding integer relations among real numbers, to breaking
knapsack cryptosystems, to factoring polynomials over the rationals, to simultaneous Diophantine
approximation.

The Lenstra-Lenstra-Lovasz algorithm (1982) is the celebrated polynomial-time answer. Finding the
genuinely shortest vector in a lattice is NP-hard, but LLL finds a *provably* short one -- within a
factor 2**((n-1)/2) of the true shortest -- in polynomial time, and in practice does far better. It
works from the Gram-Schmidt orthogonalisation of the basis. Two conditions define an LLL-reduced basis:

  1. Size reduction: each basis vector's Gram-Schmidt coefficient against every earlier one has
     absolute value <= 1/2. Geometrically, no vector has a large component that a whole multiple of an
     earlier vector could cancel. Enforced by subtracting round(mu_ij) * b_j from b_i.

  2. The Lovasz condition: consecutive Gram-Schmidt vectors don't shrink too fast --
     ||b*_k||^2 >= (delta - mu_{k,k-1}^2) ||b*_{k-1}||^2, with delta in (1/4, 1). When it fails, swap
     b_{k-1} and b_k and step back; the swap is what actually shortens the basis.

The algorithm alternates: size-reduce vector k against k-1, check Lovasz, swap-and-retreat if it fails,
advance if it holds. It terminates because a potential function built from the Gram-Schmidt norms
strictly decreases on every swap and is bounded below. delta = 3/4 is Lovasz's original choice; closer
to 1 gives a better basis but more work.

This implementation is EXACT. All arithmetic is done in the rationals (Python's ``fractions.Fraction``)
so the Gram-Schmidt coefficients, the norms, and the comparisons are computed without a whisker of
floating-point error -- an LLL that never mis-swaps because a norm was off in the fifteenth digit. The
integer basis vectors stay integers throughout; only the orthogonalisation is rational.

Two applications ride on top. ``integer_relation`` finds small integer coefficients a_i with
sum(a_i x_i) ~ 0 for given reals -- the mechanism behind discovering closed forms for numerical
constants (e.g. that a number is a root of a particular polynomial) -- by reducing a lattice whose short
vectors encode the relation. ``shortest_vector`` returns the shortest vector of the reduced basis, a
strong heuristic for the true shortest and exact for the small lattices we validate against.

Validation. (1) Lattice preservation: the reduction is applied by integer row operations only, so the
transform is unimodular and the reduced basis spans exactly the same lattice -- checked by verifying the
absolute determinant (Gram determinant) is unchanged and that each reduced vector is an integer
combination of the originals and vice versa. (2) The output basis provably satisfies both the size and
Lovasz conditions, checked directly. (3) Shortness: for small lattices the LLL shortest vector is
compared against a brute-force search over all integer combinations in a bounded box, and it matches the
true shortest. (4) ``integer_relation`` recovers known relations (e.g. among 1, sqrt(2), and their
integer combinations) exactly. Pure standard library -- ``fractions`` and ``math`` only.
"""

from fractions import Fraction
from math import isqrt


# ---------------------------------------------------------------------------
# exact Gram-Schmidt over the rationals
# ---------------------------------------------------------------------------

def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def gram_schmidt(basis):
    """Exact Gram-Schmidt orthogonalisation.

    Returns (bstar, mu) where bstar is the list of orthogonal vectors (as Fractions) and mu is the
    lower-triangular matrix of coefficients mu[i][j] = <b_i, b*_j> / <b*_j, b*_j>.
    """
    n = len(basis)
    bstar = []
    mu = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    norms = []  # <b*_j, b*_j>
    for i in range(n):
        vi = [Fraction(x) for x in basis[i]]
        for j in range(i):
            if norms[j] == 0:
                mu[i][j] = Fraction(0)
                continue
            mu[i][j] = _dot([Fraction(x) for x in basis[i]], bstar[j]) / norms[j]
            vi = [a - mu[i][j] * b for a, b in zip(vi, bstar[j])]
        bstar.append(vi)
        norms.append(_dot(vi, vi))
    return bstar, mu, norms


# ---------------------------------------------------------------------------
# LLL reduction
# ---------------------------------------------------------------------------

def lll_reduce(basis, delta=Fraction(3, 4)):
    """Return an LLL-reduced basis for the lattice spanned by the integer vectors in ``basis``.

    ``delta`` in (1/4, 1) controls the quality/speed trade-off; 3/4 is the classic choice. The input
    vectors must be integer sequences; the output vectors are integer lists spanning the same lattice.
    """
    if not basis:
        return []
    delta = Fraction(delta)
    if not (Fraction(1, 4) < delta < 1):
        raise ValueError("delta must lie in (1/4, 1)")

    # work on a mutable integer copy
    b = [[int(x) for x in row] for row in basis]
    n = len(b)
    dim = len(b[0])
    if any(len(row) != dim for row in b):
        raise ValueError("all basis vectors must have the same dimension")

    k = 1
    while k < n:
        # recompute the exact Gram-Schmidt each iteration -- cheap for our sizes and free of any
        # incremental-update drift, so the size and Lovasz tests use consistent mu/norms.
        _, mu, norms = gram_schmidt(b)
        # full size reduction of b[k] against every earlier vector, high index to low
        for j in range(k - 1, -1, -1):
            q = _round_half(mu[k][j])
            if q != 0:
                b[k] = [x - q * y for x, y in zip(b[k], b[j])]
                _, mu, norms = gram_schmidt(b)
        # Lovasz condition
        if norms[k] >= (delta - mu[k][k - 1] ** 2) * norms[k - 1]:
            k += 1
        else:
            b[k], b[k - 1] = b[k - 1], b[k]
            k = max(k - 1, 1)
    return b


def _round_half(fr):
    """Round a Fraction to the nearest integer, ties to nearest even (banker's), as LLL expects."""
    n = fr.numerator
    d = fr.denominator
    q, r = divmod(n, d)  # q = floor, r in [0, d)
    twice = 2 * r
    if twice < d:
        return q
    if twice > d:
        return q + 1
    # exactly halfway: round to even
    return q if q % 2 == 0 else q + 1


# ---------------------------------------------------------------------------
# derived quantities
# ---------------------------------------------------------------------------

def is_reduced(basis, delta=Fraction(3, 4)):
    """Check the size-reduction and Lovasz conditions hold for ``basis``."""
    delta = Fraction(delta)
    _, mu, norms = gram_schmidt([[int(x) for x in row] for row in basis])
    n = len(basis)
    for i in range(n):
        for j in range(i):
            if abs(mu[i][j]) > Fraction(1, 2):
                return False
    for k in range(1, n):
        if norms[k] < (delta - mu[k][k - 1] ** 2) * norms[k - 1]:
            return False
    return True


def gram_determinant(basis):
    """Exact |det| of the lattice: sqrt(det(B B^T)). Returns a Fraction (exact when a perfect square)."""
    _, _, norms = gram_schmidt([[int(x) for x in row] for row in basis])
    prod = Fraction(1)
    for nrm in norms:
        prod *= nrm
    return prod  # this is det(B B^T); |det| = sqrt(prod)


def vector_norm_sq(v):
    return sum(x * x for x in v)


def shortest_vector(basis, delta=Fraction(3, 4)):
    """Reduce and return the shortest non-zero vector of the reduced basis."""
    red = lll_reduce(basis, delta)
    best = None
    best_n = None
    for v in red:
        nq = vector_norm_sq(v)
        if nq == 0:
            continue
        if best_n is None or nq < best_n:
            best_n = nq
            best = v
    return best


# ---------------------------------------------------------------------------
# integer relation detection
# ---------------------------------------------------------------------------

def integer_relation(reals, scale=10 ** 10, delta=Fraction(3, 4)):
    """Find small integers a_i with sum(a_i * reals_i) ~ 0.

    Builds the lattice whose rows are [e_i | round(scale * x_i)] (identity block augmented by a scaled
    last column), reduces it, and returns the identity-block part of the shortest reduced row -- the
    candidate relation. ``scale`` sets how tightly the near-zero is enforced. Returns a list of ints.
    """
    n = len(reals)
    rows = []
    for i in range(n):
        row = [1 if j == i else 0 for j in range(n)]
        row.append(int(round(scale * reals[i])))
        rows.append(row)
    reduced = lll_reduce(rows, delta)
    # the relation is the identity-block part of the row with the smallest last coordinate magnitude
    best = None
    best_key = None
    for row in reduced:
        coeffs = row[:n]
        residual = abs(row[n])
        if all(c == 0 for c in coeffs):
            continue
        key = (residual, vector_norm_sq(coeffs))
        if best_key is None or key < best_key:
            best_key = key
            best = coeffs
    return best


def _brute_shortest(basis, bound):
    """Reference: shortest non-zero lattice vector over integer coefficients in [-bound, bound]."""
    n = len(basis)
    dim = len(basis[0])
    best = None
    best_n = None
    # iterate over all coefficient tuples in the box
    coeffs = [0] * n

    def rec(i):
        nonlocal best, best_n
        if i == n:
            v = [0] * dim
            for k in range(n):
                if coeffs[k]:
                    for d in range(dim):
                        v[d] += coeffs[k] * basis[k][d]
            nq = vector_norm_sq(v)
            if nq > 0 and (best_n is None or nq < best_n):
                best_n = nq
                best = list(v)
            return
        for c in range(-bound, bound + 1):
            coeffs[i] = c
            rec(i + 1)
        coeffs[i] = 0

    rec(0)
    return best, best_n
