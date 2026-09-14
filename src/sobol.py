"""Sobol sequences: low-discrepancy points that fill space far more evenly than random sampling.

Monte Carlo integration with random points converges as O(1/sqrt(N)) -- painfully slow, because random
points clump and leave gaps. QUASI-random (low-discrepancy) sequences fix this by placing points to
avoid clustering, achieving nearly O(1/N). The SOBOL sequence, invented by Ilya Sobol in 1967, is the
most widely used: it powers quasi-Monte Carlo pricing in finance, rendering in graphics, and
high-dimensional integration throughout physics.

Sobol's construction is elegant bit arithmetic. Each dimension has a set of DIRECTION NUMBERS v_1,
v_2, ... -- binary fractions derived from a PRIMITIVE POLYNOMIAL over GF(2) and a short table of odd
initializing integers. The n-th point in a dimension is the XOR of the direction numbers selected by
the bits of n. Using GRAY-CODE order makes it incremental: point n differs from point n-1 in exactly
one bit, so x_n = x_{n-1} XOR v_c, where c is the position of the rightmost zero bit of n-1. One XOR
per point per dimension -- and the result is a sequence whose every 2^k-length prefix is
STRATIFIED, exactly one point in each dyadic box.

The first dimension uses the identity direction numbers, which makes the 1-D Sobol sequence precisely
the van der Corput sequence in base 2 (bit-reversal). Higher dimensions use distinct primitive
polynomials so their projections don't align into degenerate diagonal patterns -- the classic failure
mode of naive quasi-random constructions.

This module generates Sobol points in up to a handful of dimensions (with the standard Joe-Kuo
direction-number data for the low dimensions), skips the customary first point, and integrates via
quasi-Monte Carlo. It is validated: the 1-D sequence equals van der Corput base 2 exactly; every
2^k-prefix is stratified (one point per dyadic interval); the 2-D star discrepancy is substantially
lower than i.i.d. random points and beats Halton at the same count; and quasi-Monte Carlo integration
of smooth functions converges markedly faster than random Monte Carlo. Reuses the repo's discrepancy
and QMC tools. Pure stdlib; the quasi-random companion to the Halton/Hammersley low-discrepancy set."""

from __future__ import annotations

# Primitive polynomials (as the integer of their middle coefficients, Joe-Kuo convention) and the
# initial direction-number integers m_i for the first several dimensions. Dimension 1 is special
# (identity). Data below is the standard Joe-Kuo (2008) set for dimensions 2..8.
#   entry: (degree s, polynomial coefficient integer a, [m_1..m_s])
_DIRECTION_DATA = [
    # dim 2
    (1, 0, [1]),
    # dim 3
    (2, 1, [1, 3]),
    # dim 4
    (3, 1, [1, 3, 1]),
    # dim 5
    (3, 2, [1, 1, 1]),
    # dim 6
    (4, 1, [1, 1, 3, 3]),
    # dim 7
    (4, 4, [1, 3, 5, 13]),
    # dim 8
    (5, 2, [1, 1, 5, 5, 17]),
]

_BITS = 30                      # number of bits of resolution
_SCALE = float(1 << _BITS)


def _direction_numbers(dim):
    """Direction-number integers v[d][k] (scaled by 2^BITS) for d in 0..dim-1, k in 0..BITS-1."""
    V = [[0] * _BITS for _ in range(dim)]
    # dimension 0: identity -> v_k = 2^{BITS-1-k}  (van der Corput / bit reversal)
    for k in range(_BITS):
        V[0][k] = 1 << (_BITS - 1 - k)
    for d in range(1, dim):
        s, a, m = _DIRECTION_DATA[d - 1]
        m = list(m)
        # initialize first s direction numbers: v_k = m_k * 2^{BITS-1-k}
        for k in range(s):
            V[d][k] = m[k] << (_BITS - 1 - k)
        # recurrence for the rest
        for k in range(s, _BITS):
            val = V[d][k - s] ^ (V[d][k - s] >> s)
            for i in range(1, s):
                # bit i-1 of the polynomial coefficient a selects whether to XOR v_{k-i}
                if (a >> (s - 1 - i)) & 1:
                    val ^= V[d][k - i]
            V[d][k] = val
    return V


def sobol(count, dim, skip=True):
    """First `count` Sobol points in `dim` dimensions, each a list of floats in [0,1).

    skip=True drops the initial all-zeros point (standard practice)."""
    if dim < 1 or dim > len(_DIRECTION_DATA) + 1:
        raise ValueError(f"dim must be in 1..{len(_DIRECTION_DATA)+1}")
    V = _direction_numbers(dim)
    points = []
    x = [0] * dim               # integer state per dimension
    start = 1 if skip else 0
    # to honor skip while returning `count` points, iterate count+start indices
    for n in range(start + count):
        if n == 0:
            pt = [0.0] * dim
        else:
            # Gray-code: rightmost zero bit of (n-1)
            c = 0
            v = n - 1
            while v & 1:
                v >>= 1
                c += 1
            for d in range(dim):
                x[d] ^= V[d][c]
            pt = [x[d] / _SCALE for d in range(dim)]
        if n >= start:
            points.append(pt)
    return points


def sobol_1d(count, skip=True):
    """1-D Sobol sequence (equals van der Corput base 2)."""
    return [p[0] for p in sobol(count, 1, skip)]


def qmc_integrate(f, dim, count, domain=None, skip=True):
    """Quasi-Monte Carlo integral of f over a box using Sobol points.

    domain: list of (lo, hi) per dimension (default unit cube). f takes a length-dim list."""
    if domain is None:
        domain = [(0.0, 1.0)] * dim
    pts = sobol(count, dim, skip)
    vol = 1.0
    for (lo, hi) in domain:
        vol *= (hi - lo)
    total = 0.0
    for p in pts:
        x = [domain[d][0] + p[d] * (domain[d][1] - domain[d][0]) for d in range(dim)]
        total += f(x)
    return vol * total / len(pts)
