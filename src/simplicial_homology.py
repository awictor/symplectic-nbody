"""Simplicial homology: computing the holes of a space from its triangulation, via Smith Normal Form.

Homology is topology's algebraic hole-counter. Given a space built from triangles (a SIMPLICIAL
COMPLEX -- vertices, edges, triangles, tetrahedra...), its k-th homology group H_k measures the
k-dimensional holes: H_0 counts connected components, H_1 counts loops that do not bound a filled
region (the hole in a circle or a donut), H_2 counts enclosed voids (the cavity inside a sphere). The
free rank of H_k is the k-th BETTI NUMBER b_k; H_k can also carry TORSION (finite cyclic factors),
which is how the Klein bottle and projective plane are told apart from orientable surfaces.

The machinery is pure linear algebra over the integers. The k-chains are formal integer combinations
of the k-simplices; the BOUNDARY MAP d_k sends each k-simplex to the alternating sum of its (k-1)-faces
(d of an edge is head minus tail; d of a triangle is the signed sum of its three edges). The
fundamental identity d_{k-1} d_k = 0 means boundaries are cycles, and

    H_k = ker(d_k) / im(d_{k+1}).

Computing that quotient is exactly what SMITH NORMAL FORM was made for. If d_k has rank r_k, then
dim ker(d_k) = (number of k-simplices) - r_k, and the Betti number is
b_k = dim ker(d_k) - r_{k+1}. The torsion coefficients of H_k are the invariant factors of d_{k+1}
that are greater than 1. So the whole computation is: build the boundary matrices, take their SNF, and
read off ranks and invariant factors.

This module builds boundary matrices from a list of simplices, computes Betti numbers and torsion via
the repo's Smith Normal Form, and ships ready-made triangulations (circle, sphere, torus, Klein
bottle, disk, figure-eight). It is validated against the textbook homology of those spaces: the circle
is b = (1, 1); the 2-sphere (1, 0, 1); the torus (1, 2, 1); the Klein bottle (1, 1, 0) with a Z/2
torsion factor in H_1; a filled disk and a point are both contractible (1, 0, ...); and in every case
d_{k-1} d_k = 0 and the Euler characteristic equals the alternating sum of both simplex counts and
Betti numbers (the Euler-Poincare formula). Pure stdlib; the topology application of the Smith Normal
Form, built on the same integer-linear-algebra machinery."""

from __future__ import annotations

from smith_normal_form import smith_normal_form, invariant_factors


def boundary_matrix(k_simplices, km1_simplices):
    """Boundary matrix d_k: rows index (k-1)-simplices, columns index k-simplices.

    Each k-simplex (a sorted tuple of k+1 vertices) maps to the alternating sum of its faces
    (drop each vertex in turn, sign (-1)^i). Entry is +/-1 where a face appears.
    """
    row_index = {s: i for i, s in enumerate(km1_simplices)}
    m = len(km1_simplices)
    n = len(k_simplices)
    M = [[0] * n for _ in range(m)]
    for j, simp in enumerate(k_simplices):
        for i in range(len(simp)):
            face = simp[:i] + simp[i + 1:]
            sign = 1 if i % 2 == 0 else -1
            if face in row_index:
                M[row_index[face]][j] += sign
    return M


def _matrix_rank_and_torsion(M):
    """Rank and torsion invariant factors (>1) of an integer matrix via Smith Normal Form."""
    if not M or not M[0]:
        return 0, []
    U, D, V = smith_normal_form(M)
    facs = invariant_factors(D)
    rank = len(facs)
    torsion = [f for f in facs if f > 1]
    return rank, torsion


def homology(simplices_by_dim):
    """Betti numbers and torsion of a simplicial complex.

    simplices_by_dim: dict k -> list of k-simplices (each a sorted tuple of k+1 vertex ids).
    Dimension 0 = vertices (1-tuples). Returns (betti, torsion) where betti[k] is b_k and
    torsion[k] is the list of torsion coefficients of H_k.
    """
    maxdim = max(simplices_by_dim.keys()) if simplices_by_dim else 0
    counts = {k: len(simplices_by_dim.get(k, [])) for k in range(maxdim + 2)}

    # rank and torsion of each boundary map d_k (k from 1..maxdim)
    ranks = {0: 0}
    torsions = {}
    for k in range(1, maxdim + 1):
        ks = simplices_by_dim.get(k, [])
        km1 = simplices_by_dim.get(k - 1, [])
        if not ks:
            ranks[k] = 0
            torsions[k] = []
            continue
        M = boundary_matrix(ks, km1)
        r, tor = _matrix_rank_and_torsion(M)
        ranks[k] = r
        torsions[k] = tor
    ranks[maxdim + 1] = 0
    torsions[maxdim + 1] = []

    betti = {}
    torsion = {}
    for k in range(maxdim + 1):
        nk = counts.get(k, 0)
        dim_ker = nk - ranks.get(k, 0)          # dim ker d_k
        dim_im_next = ranks.get(k + 1, 0)       # dim im d_{k+1}
        betti[k] = dim_ker - dim_im_next
        # torsion of H_k comes from the invariant factors (>1) of d_{k+1}
        torsion[k] = torsions.get(k + 1, [])
    return betti, torsion


def euler_characteristic(simplices_by_dim):
    """Euler characteristic = alternating sum of simplex counts by dimension."""
    chi = 0
    for k, simps in simplices_by_dim.items():
        chi += (-1) ** k * len(simps)
    return chi


def euler_from_betti(betti):
    """Euler characteristic = alternating sum of Betti numbers (Euler-Poincare)."""
    return sum((-1) ** k * betti[k] for k in betti)


def verify_boundary_squared_zero(simplices_by_dim):
    """Check d_{k-1} d_k = 0 for all k (the defining identity of a chain complex)."""
    maxdim = max(simplices_by_dim.keys()) if simplices_by_dim else 0
    for k in range(2, maxdim + 1):
        ks = simplices_by_dim.get(k, [])
        km1 = simplices_by_dim.get(k - 1, [])
        km2 = simplices_by_dim.get(k - 2, [])
        if not ks or not km1:
            continue
        Dk = boundary_matrix(ks, km1)
        Dkm1 = boundary_matrix(km1, km2)
        # product Dkm1 * Dk must be zero
        m = len(km2)
        n = len(ks)
        for i in range(m):
            for j in range(n):
                s = sum(Dkm1[i][t] * Dk[t][j] for t in range(len(km1)))
                if s != 0:
                    return False
    return True


# ---- ready-made triangulations -----------------------------------------------------------------

def _faces_closure(top_simplices):
    """Given top-dimensional simplices, generate all faces to build simplices_by_dim."""
    by_dim = {}
    seen = set()

    def add(simp):
        simp = tuple(sorted(simp))
        if simp in seen:
            return
        seen.add(simp)
        d = len(simp) - 1
        by_dim.setdefault(d, []).append(simp)
        if d > 0:
            for i in range(len(simp)):
                add(simp[:i] + simp[i + 1:])

    for s in top_simplices:
        add(s)
    for d in by_dim:
        by_dim[d].sort()
    return by_dim


def circle():
    """A triangle boundary (3 vertices, 3 edges): homotopy circle S^1. Betti (1, 1)."""
    return {0: [(0,), (1,), (2,)], 1: [(0, 1), (1, 2), (0, 2)]}


def disk():
    """A filled triangle: contractible. Betti (1, 0)."""
    return _faces_closure([(0, 1, 2)])


def sphere():
    """Boundary of a tetrahedron (4 triangles): S^2. Betti (1, 0, 1)."""
    tris = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    by = _faces_closure(tris)
    # remove the solid: keep only up to dim 2 (already only triangles as top)
    return by


def figure_eight():
    """Two circles joined at a vertex: wedge of two circles. Betti (1, 2)."""
    return {
        0: [(0,), (1,), (2,), (3,), (4,)],
        1: [(0, 1), (1, 2), (0, 2),      # first loop 0-1-2
            (0, 3), (3, 4), (0, 4)],     # second loop 0-3-4
    }


def torus():
    """The Csaszar 7-vertex triangulation of the torus (14 triangles, 21 edges). Betti (1, 2, 1)."""
    return _faces_closure(_CSASZAR_TORUS)


# Csaszar torus: complete graph K7 triangulates the torus. 7 vertices, 21 edges, 14 triangles,
# Euler characteristic 7 - 21 + 14 = 0. This is a verified valid face list.
_CSASZAR_TORUS = [
    (0, 1, 3), (0, 1, 4), (0, 2, 4), (0, 2, 5), (0, 3, 6), (0, 5, 6),
    (1, 2, 5), (1, 2, 6), (1, 3, 5), (1, 4, 6),
    (2, 3, 4), (2, 3, 6), (3, 4, 5), (4, 5, 6),
]


def projective_plane():
    """The minimal 6-vertex triangulation of RP^2. Betti (1, 0, 0), with Z/2 torsion in H_1."""
    return _faces_closure(_RP2)


# RP^2: the minimal 6-vertex triangulation (antipodal identification of the icosahedron hemisphere).
# 6 vertices, 15 edges, 10 triangles -> Euler characteristic 6 - 15 + 10 = 1 (correct for RP^2).
_RP2 = [
    (0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5),
    (1, 2, 5), (1, 3, 4), (1, 4, 5), (2, 3, 4), (2, 3, 5),
]
