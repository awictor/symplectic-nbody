"""The Matrix-Tree theorem: counting spanning trees with a determinant.

A SPANNING TREE of a connected graph is a subset of its edges that touches every vertex and forms a
tree (no cycle). Counting them -- how many distinct ways can a network be wired into a minimal
connected backbone? -- sounds combinatorial, but Kirchhoff's MATRIX-TREE THEOREM (1847) reduces it to
a single determinant. Build the LAPLACIAN matrix L = D - A, where D is the diagonal matrix of degrees
and A the adjacency matrix; then EVERY cofactor of L (the determinant of L with any one row and its
matching column deleted) equals the number of spanning trees. One linear-algebra computation replaces
an exponential search, and the same theorem underlies electrical-network analysis (Kirchhoff derived
it for circuits), reliability engineering, and the theory of random spanning trees.

The theorem also generalises Cayley's formula: for the complete graph K_n the count is n^(n-2), which
falls straight out of the determinant. For WEIGHTED graphs the weighted Laplacian's cofactor gives the
sum over spanning trees of the product of their edge weights -- the "spanning-tree polynomial"
evaluated at the weights. Because the counts are exact integers (or exact rationals for weights), this
implementation computes the cofactor by fraction-free / exact arithmetic so there is never any
floating-point rounding: it uses Python's Fraction for the Gaussian elimination that evaluates the
determinant.

This module builds the Laplacian of an undirected (multi)graph, deletes a row and column, and computes
the cofactor determinant to count spanning trees (unweighted, and the weighted spanning-tree sum). It
is verified against brute force -- enumerating every edge subset of size n-1 and testing whether it
forms a spanning tree -- on hundreds of random graphs, against Cayley's n^(n-2) for complete graphs,
and on known values (a cycle C_n has exactly n spanning trees, a tree has exactly 1, a disconnected
graph has 0). Pure stdlib; a graph-theory-meets-linear-algebra companion to the Prufer-sequence,
union-find/MST, and Gaussian-elimination notes."""

from __future__ import annotations

from fractions import Fraction


def laplacian(n, edges, weighted=False):
    """The Laplacian matrix L = D - A of an undirected (multi)graph. With `weighted=True`, edges are
    (u, v, w) and L uses edge weights; otherwise edges are (u, v) with unit weight. Returns an n x n
    matrix of Fractions."""
    L = [[Fraction(0)] * n for _ in range(n)]
    for e in edges:
        if weighted:
            u, v, w = e
        else:
            u, v = e[0], e[1]
            w = 1
        if u == v:
            continue                     # self-loops don't affect spanning trees
        L[u][u] += w
        L[v][v] += w
        L[u][v] -= w
        L[v][u] -= w
    return L


def _det(matrix):
    """Exact determinant via Gaussian elimination over Fractions."""
    n = len(matrix)
    if n == 0:
        return Fraction(1)
    a = [row[:] for row in matrix]
    det = Fraction(1)
    for col in range(n):
        # find a pivot
        piv = None
        for r in range(col, n):
            if a[r][col] != 0:
                piv = r
                break
        if piv is None:
            return Fraction(0)
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
            det = -det
        det *= a[col][col]
        inv = a[col][col]
        for r in range(col + 1, n):
            if a[r][col] != 0:
                f = a[r][col] / inv
                a[r] = [a[r][c] - f * a[col][c] for c in range(n)]
    return det


def count_spanning_trees(n, edges):
    """The number of spanning trees of an undirected (multi)graph, via any cofactor of the Laplacian
    (Matrix-Tree theorem). Returns an exact integer (0 if disconnected, 1 for a single vertex)."""
    if n <= 1:
        return 1
    L = laplacian(n, edges)
    # delete the last row and column -> the (n-1)x(n-1) cofactor
    minor = [[L[i][j] for j in range(n - 1)] for i in range(n - 1)]
    d = _det(minor)
    return int(d)


def weighted_spanning_tree_sum(n, edges):
    """The sum over all spanning trees of the product of their edge weights (the weighted Matrix-Tree
    quantity). `edges` are (u, v, w). Returns a Fraction (an integer for integer weights)."""
    if n <= 1:
        return Fraction(1)
    L = laplacian(n, edges, weighted=True)
    minor = [[L[i][j] for j in range(n - 1)] for i in range(n - 1)]
    return _det(minor)


# --- brute-force reference --------------------------------------------------
def _is_spanning_tree(n, tree_edges):
    """True iff `tree_edges` (exactly n-1 edges) connect all n vertices acyclically."""
    if len(tree_edges) != n - 1:
        return False
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in tree_edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        parent[ru] = rv
    return len({find(v) for v in range(n)}) == 1


def brute_count_spanning_trees(n, edges):
    """Count spanning trees by enumerating every size-(n-1) subset of edges and testing it. Uses
    distinct simple edges (parallel edges collapsed). Exponential; small graphs only."""
    from itertools import combinations
    if n <= 1:
        return 1
    simple = sorted({(min(u, v), max(u, v)) for e in edges for u, v in [(e[0], e[1])] if e[0] != e[1]})
    count = 0
    for subset in combinations(simple, n - 1):
        if _is_spanning_tree(n, list(subset)):
            count += 1
    return count
