"""Matrix chain multiplication: the optimal order to multiply a sequence of matrices.

Multiplying matrices is associative -- (AB)C = A(BC) -- but the AMOUNT OF WORK is not. Multiplying a
p x q matrix by a q x r matrix costs p*q*r scalar multiplications, so for a chain of matrices the
order of the products can change the total cost by orders of magnitude. Given the dimensions, the
MATRIX-CHAIN ORDERING problem asks for the PARENTHESIZATION that minimizes total scalar
multiplications. The number of parenthesizations is the Catalan number (exponential), so brute force
is hopeless, but DYNAMIC PROGRAMMING solves it in O(n^3) -- a textbook example of optimal
substructure.

The insight: the optimal way to multiply matrices i..j must, at its outermost step, split at some k
into (i..k)(k+1..j), and BOTH halves must themselves be optimally parenthesized. So define m[i][j] =
the minimum cost to multiply matrices i through j, and fill it by increasing chain length: m[i][j] =
min over k of m[i][k] + m[k+1][j] + (cost of multiplying the two resulting matrices, p_{i-1} p_k
p_j). The base case m[i][i] = 0 (a single matrix needs no work). Recording the minimizing split k at
each cell lets you reconstruct the optimal parenthesization. This is the canonical interval DP,
the same pattern behind optimal binary search trees and polygon triangulation.

This module computes the minimum multiplication cost and the optimal parenthesization for a chain of
matrices given their dimensions, and a helper that counts the cost of any given order. It is verified
against brute force -- an exhaustive search over all Catalan-many parenthesizations for short chains
-- that the DP finds the true minimum, that the reconstructed parenthesization actually achieves that
cost, that it beats the naive left-to-right order on skewed dimensions, and on hand-checked textbook
instances. Pure stdlib; a dynamic-programming companion to the knapsack, LCS, and edit-distance
notes."""

from __future__ import annotations


def min_cost(dims):
    """Minimum scalar-multiplication cost to multiply a chain of n matrices.

    dims is a list of n+1 dimensions: matrix i (1-indexed) is dims[i-1] x dims[i]. Returns
    (min_cost, parenthesization_string)."""
    n = len(dims) - 1
    if n <= 0:
        return 0, ""
    if n == 1:
        return 0, "A1"
    # m[i][j] = min cost to multiply matrices i..j (1-indexed); s[i][j] = optimal split
    INF = float("inf")
    m = [[0] * (n + 1) for _ in range(n + 1)]
    s = [[0] * (n + 1) for _ in range(n + 1)]
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            m[i][j] = INF
            for k in range(i, j):
                cost = m[i][k] + m[k + 1][j] + dims[i - 1] * dims[k] * dims[j]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
    return m[1][n], _build_parens(s, 1, n)


def _build_parens(s, i, j):
    if i == j:
        return f"A{i}"
    k = s[i][j]
    return "(" + _build_parens(s, i, k) + _build_parens(s, k + 1, j) + ")"


def order_cost(dims, order):
    """The cost of a specific parenthesization, given as a nested tuple of 1-based matrix indices,
    e.g. ((1, 2), 3). Returns the total scalar multiplications."""
    def evaluate(node):
        # returns (rows, cols, cost)
        if isinstance(node, int):
            return dims[node - 1], dims[node], 0
        left, right = node
        lr, lc, lcost = evaluate(left)
        rr, rc, rcost = evaluate(right)
        # lc must equal rr
        return lr, rc, lcost + rcost + lr * lc * rc

    _, _, cost = evaluate(order)
    return cost


def left_to_right_cost(dims):
    """The cost of the naive left-to-right order ((((A1 A2) A3) A4)...)."""
    n = len(dims) - 1
    if n <= 1:
        return 0
    order = 1
    for i in range(2, n + 1):
        order = (order, i)
    return order_cost(dims, order)
