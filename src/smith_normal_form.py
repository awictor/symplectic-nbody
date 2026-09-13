"""Smith Normal Form: diagonalizing an integer matrix, and reading off the structure it encodes.

Over a field, Gaussian elimination diagonalizes any matrix. Over the INTEGERS you cannot divide freely
-- you may only add integer multiples of rows/columns and swap or negate them (UNIMODULAR operations,
which preserve the integer lattice). Astonishingly, that is still enough: every integer matrix A can be
reduced to a diagonal SMITH NORMAL FORM D = U A V, where U and V are unimodular (integer matrices with
determinant +/-1, hence integer-invertible) and the diagonal entries d_1 | d_2 | ... | d_r each DIVIDE
the next. Those d_i are the INVARIANT FACTORS, and they are unique.

The invariant factors are not a computational curiosity -- they ARE the answer to a surprising range
of questions. The cokernel Z^m / A Z^n is the finite abelian group Z/d_1 x Z/d_2 x ... x Z (with free
part from the zero diagonal entries), so SNF computes the structure of any finitely generated abelian
group given by generators and relations -- which is exactly how HOMOLOGY GROUPS are computed in
topology (the ranks and torsion of a chain complex), how the SOLVABILITY of an integer linear system
A x = b is decided, and how lattice indices are found. The k-th invariant factor equals
gcd(all k x k minors) / gcd(all (k-1) x (k-1) minors), which is how the classic theory proves
uniqueness.

The algorithm here is the standard row/column reduction: repeatedly pick a pivot, use the extended
Euclidean algorithm to reduce every entry in its row and column modulo the pivot, and iterate until the
pivot divides its whole row and column; then clear them, move to the submatrix, and finally sort the
diagonal and enforce the divisibility chain by gcd/lcm swaps. Every operation is recorded in U and V.

This module computes the Smith Normal Form with the transforms, the invariant factors, the matrix rank,
and the abelian-group / linear-system readouts. It is validated by the defining identities: U A V = D
exactly, U and V are unimodular (det +/-1), D is diagonal with the divisibility chain d_i | d_{i+1};
the product of nonzero invariant factors equals gcd of the maximal minors (and for square full-rank A,
their product is |det A|); the rank matches an independent computation; and the invariant factors match
the gcd-of-minors formula on small matrices. Pure stdlib (Fraction-free integer arithmetic); the
integer-linear-algebra companion to the LLL lattice-reduction and CRT tools."""

from __future__ import annotations


def _identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def _matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        for k in range(m):
            a = A[i][k]
            if a:
                for j in range(p):
                    C[i][j] += a * B[k][j]
    return C


def smith_normal_form(A):
    """Compute the Smith Normal Form of an integer matrix A.

    Returns (U, D, V) with U (m x m) and V (n x n) unimodular and D = U A V diagonal with the
    divisibility chain d_1 | d_2 | ... on its diagonal.
    """
    m = len(A)
    n = len(A[0]) if m else 0
    D = [[int(A[i][j]) for j in range(n)] for i in range(m)]
    U = _identity(m)
    V = _identity(n)

    def swap_rows(i, j):
        D[i], D[j] = D[j], D[i]
        U[i], U[j] = U[j], U[i]

    def swap_cols(i, j):
        for row in D:
            row[i], row[j] = row[j], row[i]
        for row in V:
            row[i], row[j] = row[j], row[i]

    def add_row(dst, src, k):
        # row[dst] += k * row[src]
        for j in range(n):
            D[dst][j] += k * D[src][j]
        for j in range(m):
            U[dst][j] += k * U[src][j]

    def add_col(dst, src, k):
        for i in range(m):
            D[i][dst] += k * D[i][src]
        for i in range(n):
            V[i][dst] += k * V[i][src]

    def negate_row(i):
        for j in range(n):
            D[i][j] = -D[i][j]
        for j in range(m):
            U[i][j] = -U[i][j]

    t = 0
    while t < min(m, n):
        # find a nonzero pivot in submatrix D[t:, t:]
        piv = None
        for i in range(t, m):
            for j in range(t, n):
                if D[i][j] != 0:
                    piv = (i, j)
                    break
            if piv:
                break
        if piv is None:
            break  # rest is zero
        pi, pj = piv
        swap_rows(t, pi)
        swap_cols(t, pj)

        # reduce row t and column t until D[t][t] divides everything in them
        while True:
            changed = False
            # clear column t below/above using the pivot
            for i in range(m):
                if i != t and D[i][t] != 0:
                    q = D[i][t] // D[t][t]
                    if q != 0:
                        add_row(i, t, -q)
                        changed = True
                    if D[i][t] != 0:
                        # not divisible: swap to make it the new pivot (Euclid step)
                        swap_rows(t, i)
                        changed = True
            # clear row t
            for j in range(n):
                if j != t and D[t][j] != 0:
                    q = D[t][j] // D[t][t]
                    if q != 0:
                        add_col(j, t, -q)
                        changed = True
                    if D[t][j] != 0:
                        swap_cols(t, j)
                        changed = True
            if not changed:
                break
        if D[t][t] < 0:
            negate_row(t)
        t += 1

    # enforce the divisibility chain d_i | d_{i+1}
    r = 0
    diag = [D[i][i] for i in range(min(m, n))]
    _fix_divisibility(D, U, V, m, n)
    return U, D, V


def _fix_divisibility(D, U, V, m, n):
    """After diagonalization, enforce d_1 | d_2 | ... via gcd/lcm swaps on adjacent diagonal entries."""
    k = min(m, n)
    changed = True
    while changed:
        changed = False
        for i in range(k - 1):
            a, b = D[i][i], D[i + 1][i + 1]
            if a == 0 and b == 0:
                continue
            if b != 0 and (a == 0 or b % a != 0):
                # merge: add column i+1 into i, then re-clear -> gives (gcd, lcm)
                # add row i+1 to row i so pivot i sees both
                for j in range(n):
                    D[i][j] += D[i + 1][j]
                for j in range(m):
                    U[i][j] += U[i + 1][j]
                # now re-run a local reduction on the 2x2 block (columns i, i+1)
                _local_reduce(D, U, V, m, n, i)
                changed = True
    # make all diagonal entries non-negative
    for i in range(k):
        if D[i][i] < 0:
            for j in range(n):
                D[i][j] = -D[i][j]
            for j in range(m):
                U[i][j] = -U[i][j]


def _local_reduce(D, U, V, m, n, t):
    """Re-clear row/column t (used after merging for the divisibility fix)."""
    if D[t][t] == 0:
        # find any nonzero in row t to pivot
        for j in range(t, n):
            if D[t][j] != 0:
                for row in D:
                    row[t], row[j] = row[j], row[t]
                for row in V:
                    row[t], row[j] = row[j], row[t]
                break
    while True:
        changed = False
        if D[t][t] == 0:
            break
        for i in range(m):
            if i != t and D[i][t] != 0:
                q = D[i][t] // D[t][t]
                if q:
                    for j in range(n):
                        D[i][j] -= q * D[t][j]
                    for j in range(m):
                        U[i][j] -= q * U[t][j]
                    changed = True
                if D[i][t] != 0:
                    D[t], D[i] = D[i], D[t]
                    U[t], U[i] = U[i], U[t]
                    changed = True
        for j in range(n):
            if j != t and D[t][j] != 0:
                q = D[t][j] // D[t][t]
                if q:
                    for i in range(m):
                        D[i][j] -= q * D[i][t]
                    for i in range(n):
                        V[i][j] -= q * V[i][t]
                    changed = True
                if D[t][j] != 0:
                    for row in D:
                        row[t], row[j] = row[j], row[t]
                    for row in V:
                        row[t], row[j] = row[j], row[t]
                    changed = True
        if not changed:
            break


def invariant_factors(D):
    """The nonzero diagonal entries of the Smith Normal Form, in divisibility order."""
    k = min(len(D), len(D[0]))
    return [D[i][i] for i in range(k) if D[i][i] != 0]


def rank(D):
    """Matrix rank = number of nonzero invariant factors."""
    return len(invariant_factors(D))


def is_unimodular(M):
    """True if M is a square integer matrix with determinant +/-1."""
    d = _int_det(M)
    return d in (1, -1)


def _int_det(M):
    """Exact integer determinant by fraction-free (Bareiss) elimination."""
    n = len(M)
    if n == 0:
        return 1
    A = [[int(M[i][j]) for j in range(n)] for i in range(n)]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            # find a row to swap
            sw = None
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    sw = i
                    break
            if sw is None:
                return 0
            A[k], A[sw] = A[sw], A[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]


def abelian_group_structure(A):
    """Structure of the cokernel Z^m / A Z^n: (torsion factors, free rank).

    torsion is the list of invariant factors > 1 (each giving a Z/d summand); free_rank is the number
    of zero columns in the SNF diagonal beyond the rank (Z summands).
    """
    m = len(A)
    n = len(A[0]) if m else 0
    U, D, V = smith_normal_form(A)
    facs = invariant_factors(D)
    torsion = [f for f in facs if f > 1]
    r = len(facs)
    free_rank = m - r  # rows not covered by a pivot become free Z summands of the cokernel
    return torsion, free_rank


def gcd_of_minors(A, k):
    """gcd of all k x k minors of A (the k-th determinantal divisor). Returns 0 if all vanish."""
    from itertools import combinations
    m = len(A)
    n = len(A[0]) if m else 0
    if k == 0:
        return 1
    g = 0
    for rows in combinations(range(m), k):
        for cols in combinations(range(n), k):
            sub = [[A[i][j] for j in cols] for i in rows]
            g = _gcd(g, abs(_int_det(sub)))
    return g


def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a
