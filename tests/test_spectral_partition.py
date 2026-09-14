"""Tests for spectral partitioning: Laplacian properties, component count vs BFS, Fiedler bisection, brute-force min-cut."""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import spectral_partition as SP  # noqa: E402
from jacobi_eigen import sorted_eigen  # noqa: E402


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _clique(base, k):
    return [(base + i, base + j) for i in range(k) for j in range(i + 1, k)]


def _brute_min_bisection(n, edges):
    """Minimum cut over all balanced bipartitions (n even). Exponential -- only for tiny n."""
    best = None
    half = n // 2
    verts = list(range(n))
    for combo in itertools.combinations(verts[1:], half - 1):  # fix vertex 0 in side A
        side_a = set(combo) | {0}
        labels = [0 if i in side_a else 1 for i in range(n)]
        c = SP.cut_size(edges, labels)
        if best is None or c < best:
            best = c
    return best


def main():
    # ---- 1. Laplacian rows sum to zero, symmetric ---------------------------------------
    edges = _clique(0, 4) + _clique(4, 4) + [(3, 4)]
    n = 8
    L = SP.laplacian(n, edges)
    check("Laplacian rows sum to zero", all(abs(sum(row)) < 1e-9 for row in L))
    check("Laplacian symmetric", all(abs(L[i][j] - L[j][i]) < 1e-12 for i in range(n) for j in range(n)))

    # ---- 2. smallest eigenvalue is 0 with a constant eigenvector ------------------------
    vals, vecs = sorted_eigen(L)  # descending
    check("smallest Laplacian eigenvalue is 0", abs(vals[-1]) < 1e-8, f"{vals[-1]}")
    const_vec = [vecs[i][-1] for i in range(n)]
    # constant vector => all entries equal
    check("null eigenvector is constant",
          max(const_vec) - min(const_vec) < 1e-7, f"spread {max(const_vec)-min(const_vec)}")
    check("all eigenvalues >= 0 (PSD)", all(v > -1e-8 for v in vals), f"min {min(vals)}")

    # ---- 3. zero-eigenvalue count == connected components (vs BFS) ----------------------
    # connected: 1 component, 1 zero eigenvalue
    ncomp, _ = SP.connected_components(n, edges)
    check("connected: 1 component", ncomp == 1)
    check("connected: 1 zero eigenvalue", SP.count_zero_eigenvalues(n, edges) == 1)
    check("connected: Fiedler value > 0", SP.fiedler(n, edges)[0] > 1e-6)

    # disconnected: two separate cliques
    ed2 = _clique(0, 4) + _clique(4, 4)
    c2, _ = SP.connected_components(n, ed2)
    check("disconnected: 2 components (BFS)", c2 == 2)
    check("disconnected: 2 zero eigenvalues", SP.count_zero_eigenvalues(n, ed2) == 2)
    check("disconnected: Fiedler value ~ 0", abs(SP.fiedler(n, ed2)[0]) < 1e-6)

    # three components
    ed3 = _clique(0, 3) + _clique(3, 3) + _clique(6, 3)
    c3, _ = SP.connected_components(9, ed3)
    check("three cliques: 3 components", c3 == 3)
    check("three cliques: 3 zero eigenvalues", SP.count_zero_eigenvalues(9, ed3) == 3)

    # ---- 4. Fiedler sign partition recovers planted two clusters ------------------------
    labels = SP.partition(n, edges)
    # the two cliques {0,1,2,3} and {4,5,6,7} should end up on opposite sides
    check("clique 0-3 same label", len(set(labels[0:4])) == 1, f"{labels[0:4]}")
    check("clique 4-7 same label", len(set(labels[4:8])) == 1, f"{labels[4:8]}")
    check("two cliques on opposite sides", labels[0] != labels[4])
    check("cut only the bridge edge", SP.cut_size(edges, labels) == 1.0,
          f"{SP.cut_size(edges, labels)}")

    # ---- 5. spectral cut matches brute-force minimum bisection (small graph) ------------
    # planted two-community graph on 10 vertices with a couple of bridges
    e = _clique(0, 5) + _clique(5, 5) + [(4, 5), (0, 9)]
    spec_labels = SP.partition(10, e)
    spec_cut = SP.cut_size(e, spec_labels)
    brute = _brute_min_bisection(10, e)
    check("spectral cut == brute-force min bisection", abs(spec_cut - brute) < 1e-9,
          f"spectral {spec_cut} vs brute {brute}")

    # ---- 6. path graph: Fiedler vector is monotone (orders the path) --------------------
    pn = 8
    pedges = [(i, i + 1) for i in range(pn - 1)]
    fval, fvec = SP.fiedler(pn, pedges)
    # the Fiedler vector of a path is monotone along the path (up to global sign)
    diffs = [fvec[i + 1] - fvec[i] for i in range(pn - 1)]
    monotone = all(d > 1e-9 for d in diffs) or all(d < -1e-9 for d in diffs)
    check("path graph Fiedler vector monotone", monotone)
    check("path graph connected (Fiedler > 0)", fval > 1e-6)

    # ---- 7. ratio cut and normalized cut are finite and positive for a real bisection ---
    rc = SP.ratio_cut(n, edges, labels)
    nc = SP.normalized_cut(n, edges, labels)
    check("ratio cut finite positive", 0 < rc < float("inf"), f"{rc}")
    check("normalized cut finite positive", 0 < nc < float("inf"), f"{nc}")
    # a degenerate all-one-side partition has infinite ratio cut
    check("degenerate partition -> inf ratio cut", SP.ratio_cut(n, edges, [0] * n) == float("inf"))

    # ---- 8. weighted edges: heavy edges kept out of the cut -----------------------------
    # two heavy-bonded pairs joined by a light bridge; balanced bisection keeps pairs together
    we = [(0, 1), (2, 3), (1, 2)]
    ww = [10.0, 10.0, 1.0]
    wl = SP.partition(4, we, ww)
    check("weighted: heavy pair (0,1) together", wl[0] == wl[1], f"{wl}")
    check("weighted: heavy pair (2,3) together", wl[2] == wl[3], f"{wl}")
    check("weighted: only the light bridge is cut", abs(SP.cut_size(we, wl, ww) - 1.0) < 1e-9,
          f"cut {SP.cut_size(we, wl, ww)}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
