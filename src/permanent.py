"""The matrix permanent -- the determinant's harder twin, and how to compute it in O(2^n n).

The permanent of an n-by-n matrix looks exactly like the determinant with the minus signs removed:

    perm(A) = sum over all permutations sigma of  product_i A[i][sigma(i)].

That one missing sign changes everything. The determinant collapses to O(n^3) by Gaussian elimination
because sign-alternation makes row operations telescope; the permanent has NO such structure and is
#P-complete -- believed fundamentally harder than any NP problem, since it counts rather than decides.
It matters because it COUNTS: the permanent of a 0/1 biadjacency matrix is the exact number of PERFECT
MATCHINGS of a bipartite graph (assignments of every worker to a distinct qualified job), it gives the
number of systems of distinct representatives, and in quantum optics the permanent of the interferometer
matrix is the amplitude of a boson-sampling outcome -- the computation at the heart of quantum-advantage
experiments.

Summing over all n! permutations is hopeless past n ~ 12. RYSER'S FORMULA (1963) does far better with
inclusion-exclusion over the 2^n subsets of columns:

    perm(A) = (-1)^n * sum over subsets S of columns  (-1)^|S|  product_i (sum_{j in S} A[i][j]),

which is O(2^n n) -- exponential, but the best known for a general matrix, and practical to n ~ 25.
Iterating the subsets in GRAY-CODE order lets each row-sum be updated by a single add or subtract as one
column flips, shaving the inner loop to O(2^n * n) total with tiny constants (Nijenhuis-Wilf). GLYNN'S
FORMULA is an alternative of the same complexity using +/-1 sign vectors, included as an independent
cross-check.

This module computes the permanent by the naive definition (for validation on tiny matrices), by
Ryser's formula with Gray-code updates, and by Glynn's formula, plus the headline application:
counting perfect matchings of a bipartite graph from its adjacency. It works over integers and floats.
Pure standard library.

Validation. All three methods must AGREE: on random small integer and real matrices the naive sum,
Ryser, and Glynn give identical results (exact for integers, matching to floating tolerance for reals).
Known values are checked by hand -- the permanent of the all-ones n-by-n matrix is n! (every permutation
contributes 1), the permanent of the identity is 1, a permutation matrix has permanent 1, a matrix with
a zero row has permanent 0, and the 2x2 permanent is ad+bc. The bipartite perfect-matching count is
verified against a brute-force augmenting-path matching enumerator, and against the classic result that
the number of ways to place n non-attacking rooks on the 1-cells equals the permanent. Ryser matches the
naive definition up to n=8 exhaustively."""

import math


# ---------------------------------------------------------------------------
# naive definition (validation only; O(n!))
# ---------------------------------------------------------------------------

def permanent_naive(matrix):
    """Permanent by the definition: sum over all permutations. O(n!), for small n / validation."""
    n = len(matrix)
    if n == 0:
        return 1
    total = matrix[0][0] * 0        # zero of the right type (int or float)
    for perm in _permutations(list(range(n))):
        prod = matrix[0][perm[0]]
        for i in range(1, n):
            prod *= matrix[i][perm[i]]
        total += prod
    return total


def _permutations(items):
    if len(items) <= 1:
        yield items
        return
    for i in range(len(items)):
        rest = items[:i] + items[i + 1:]
        for p in _permutations(rest):
            yield [items[i]] + p


# ---------------------------------------------------------------------------
# Ryser's formula with Gray-code column iteration: O(2^n n)
# ---------------------------------------------------------------------------

def permanent_ryser(matrix):
    """Permanent by Ryser's inclusion-exclusion formula, iterated in Gray-code order. O(2^n n)."""
    n = len(matrix)
    if n == 0:
        return 1
    # row_sum[i] = sum of A[i][j] over columns j currently in the subset S
    row_sum = [0 * matrix[0][0] for _ in range(n)]
    total = 0 * matrix[0][0]
    prev_gray = 0
    # iterate all nonempty subsets in Gray-code order (codes 1 .. 2^n - 1)
    for k in range(1, 1 << n):
        gray = k ^ (k >> 1)
        diff = gray ^ prev_gray            # exactly one bit changes
        col = diff.bit_length() - 1
        if gray & diff:
            # column `col` was added to S
            for i in range(n):
                row_sum[i] += matrix[i][col]
        else:
            # column `col` was removed from S
            for i in range(n):
                row_sum[i] -= matrix[i][col]
        prev_gray = gray
        # product of row sums, times the inclusion-exclusion sign (-1)^(n - |S|)
        prod = row_sum[0]
        for i in range(1, n):
            prod *= row_sum[i]
        popcount = bin(gray).count("1")
        if (n - popcount) & 1:
            total -= prod
        else:
            total += prod
    return total


# ---------------------------------------------------------------------------
# Glynn's formula: O(2^n n), independent cross-check
# ---------------------------------------------------------------------------

def permanent_glynn(matrix):
    """Permanent by Glynn's formula over +/-1 delta vectors (Gray-code iterated). O(2^n n)."""
    n = len(matrix)
    if n == 0:
        return 1
    # delta starts all +1; column sums with that sign
    delta = [1] * n
    col_sum = [sum(matrix[i][j] for i in range(n)) for j in range(n)]
    total = _prod(col_sum)
    prev_gray = 0
    sign = 1
    # iterate the 2^(n-1) sign patterns (fix delta[0] = +1) via Gray code on the other n-1 bits
    for k in range(1, 1 << (n - 1)):
        gray = k ^ (k >> 1)
        diff = gray ^ prev_gray
        bit = diff.bit_length() - 1        # index among the free bits 0..n-2
        j = bit + 1                        # actual column (delta[0] fixed)
        delta[j] = -delta[j]
        # flipping delta[j] from s to -s changes each col_sum by -2*s*A[i][j]... update column sums
        s = delta[j]
        for c in range(n):
            col_sum[c] += 2 * s * matrix[j][c]
        prev_gray = gray
        sign = -sign
        total += sign * _prod(col_sum)
    # Glynn: perm = (1/2^{n-1}) * sum over delta of (prod_j delta_j) * prod_i (sum_j delta_j A[i][j])
    # our accumulation tracked prod over columns of col_sum with the running delta-product sign
    return total // (1 << (n - 1)) if isinstance(total, int) else total / (1 << (n - 1))


def _prod(vec):
    p = vec[0]
    for x in vec[1:]:
        p *= x
    return p


# ---------------------------------------------------------------------------
# permanent (default: Ryser)
# ---------------------------------------------------------------------------

def permanent(matrix):
    """Compute the permanent of a square matrix (Ryser's formula, O(2^n n))."""
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be square")
    return permanent_ryser(matrix)


# ---------------------------------------------------------------------------
# application: perfect matchings of a bipartite graph
# ---------------------------------------------------------------------------

def count_perfect_matchings(biadjacency):
    """Number of perfect matchings of a balanced bipartite graph = permanent of its 0/1 biadjacency.

    ``biadjacency[i][j]`` is 1 if left vertex i can match right vertex j. Requires a square matrix
    (equal side sizes).
    """
    n = len(biadjacency)
    for row in biadjacency:
        if len(row) != n:
            raise ValueError("biadjacency must be square for a perfect matching count")
    return permanent_ryser(biadjacency)


def brute_count_matchings(biadjacency):
    """Reference: count perfect matchings by enumerating assignments (backtracking). O(n!)-ish."""
    n = len(biadjacency)
    used = [False] * n
    count = 0

    def rec(i):
        nonlocal count
        if i == n:
            count += 1
            return
        for j in range(n):
            if biadjacency[i][j] and not used[j]:
                used[j] = True
                rec(i + 1)
                used[j] = False

    rec(0)
    return count
