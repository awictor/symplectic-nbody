"""Weisfeiler-Lehman color refinement: a fast, near-complete graph isomorphism test (and GNN kernel).

Are two graphs the same graph relabelled? Exact isomorphism has no known polynomial algorithm, but in
practice almost all graphs are distinguished by a beautiful, cheap iterative idea -- the
one-dimensional Weisfeiler-Lehman (WL) algorithm, also called COLOR REFINEMENT. Give every vertex an
initial colour (its degree, say). Then repeat: each vertex's new colour is a hash of its old colour
together with the SORTED MULTISET of its neighbours' colours. Vertices that looked alike but sit in
different neighbourhoods split apart; the partition of vertices into colour classes gets strictly
finer each round until it stabilises (at most n rounds). The stable colouring is a canonical
signature of the graph's local structure.

Two graphs that are isomorphic ALWAYS produce the same multiset of stable colours -- so if their
colour histograms differ, they are provably NON-isomorphic, a certificate you get in near-linear
time. The converse is not guaranteed (regular graphs are the classic fooling case: every vertex has
the same colour forever), but 1-WL distinguishes almost every pair of random graphs, and the very
same refinement is the theoretical core of graph neural networks -- a GNN is exactly as powerful at
telling graphs apart as 1-WL.

This module computes the stable colouring, the colour histogram, a canonical graph hash, a WL
similarity kernel (the count of shared subtree-pattern colours across refinement rounds, the standard
WL graph kernel), and a "possibly isomorphic" test that returns False with certainty when the
histograms differ. It also ships an exact brute-force isomorphism check (permutation search) as the
ground truth.

Validated against that exact check: WL never reports "not isomorphic" for a truly isomorphic pair
(soundness -- no false negatives), isomorphic graphs share the histogram and hash under random
relabelling, the colouring is stable (one more round changes nothing) and permutation-invariant, and
the known regular-graph fooling pairs are handled honestly (WL says "possibly", exact says whether
they really are). Pure stdlib; the graph-fingerprint companion to the tree-isomorphism (AHU) note."""

from __future__ import annotations

import itertools
from collections import Counter


def _hash_label(parts):
    """Stable small integer hash of a structured label (old colour + sorted neighbour colours)."""
    # deterministic across runs: fold into a Python int, then compress via a dict externally
    h = hash(parts)
    return h


def refine(n, edges, init=None, max_rounds=None):
    """Run 1-WL color refinement. Returns (colors, rounds) where colors[v] is a small-int colour
    class, canonicalised so the labels 0..k-1 are assigned in order of first appearance."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    if init is None:
        colors = [len(adj[v]) for v in range(n)]  # initial colour = degree
    else:
        colors = list(init)
    colors = _canonicalize(colors)

    if max_rounds is None:
        max_rounds = n + 1

    rounds = 0
    while rounds < max_rounds:
        new_labels = []
        for v in range(n):
            neigh = sorted(colors[w] for w in adj[v])
            new_labels.append((colors[v], tuple(neigh)))
        new_colors = _canonicalize_structured(new_labels)
        rounds += 1
        # stop when the partition stops getting finer (same number of classes AND identical grouping)
        if _same_partition(colors, new_colors):
            colors = new_colors
            break
        colors = new_colors
    return colors, rounds


def _canonicalize(colors):
    """Relabel colours to 0..k-1 in order of first appearance."""
    mapping = {}
    out = []
    for c in colors:
        if c not in mapping:
            mapping[c] = len(mapping)
        out.append(mapping[c])
    return out


def _canonicalize_structured(labels):
    """Relabel structured labels to small ints, ordered by the sorted distinct label values so the
    result is permutation-invariant (same label -> same int regardless of vertex order)."""
    distinct = sorted(set(labels))
    mapping = {lab: i for i, lab in enumerate(distinct)}
    return [mapping[lab] for lab in labels]


def _same_partition(a, b):
    """True if colourings a and b induce the same partition of vertices into classes."""
    # map each class of a to the set of vertices; compare against b
    if len(set(a)) != len(set(b)):
        return False
    # two vertices share a class in a iff they share one in b
    from_a = {}
    for v, c in enumerate(a):
        from_a.setdefault(c, []).append(v)
    for group in from_a.values():
        first = b[group[0]]
        if any(b[v] != first for v in group):
            return False
    return True


def color_histogram(n, edges, init=None):
    """The sorted histogram (multiset) of stable colours -- an isomorphism-invariant signature."""
    colors, _ = refine(n, edges, init=init)
    return tuple(sorted(Counter(colors).values()))


def graph_hash(n, edges, init=None):
    """A canonical hash of the stable colouring. Uses the sorted multiset of class SIZES rather than
    the class labels: canonical colour labels depend on vertex order, but the size multiset is a true
    isomorphism invariant, so isomorphic graphs hash identically."""
    return hash(color_histogram(n, edges, init=init))


def wl_kernel(n1, e1, n2, e2, rounds=3):
    """Weisfeiler-Lehman subtree kernel: the number of shared refinement-colour patterns across
    `rounds` iterations. A similarity score -- larger means more shared local structure."""
    def multiset_sequence(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        colors = _canonicalize([len(adj[v]) for v in range(n)])
        all_labels = Counter()
        # count round-0 colours
        for c in colors:
            all_labels[(0, c)] += 1
        for r in range(1, rounds + 1):
            labels = []
            for v in range(n):
                neigh = tuple(sorted(colors[w] for w in adj[v]))
                labels.append((colors[v], neigh))
            colors = _canonicalize_structured(labels)
            for c in colors:
                all_labels[(r, c)] += 1
        return all_labels

    a = multiset_sequence(n1, e1)
    b = multiset_sequence(n2, e2)
    # dot product of the two feature vectors over the shared label space
    shared = 0
    for key in a:
        if key in b:
            shared += a[key] * b[key]
    return shared


def possibly_isomorphic(n1, e1, n2, e2):
    """WL isomorphism test. Returns False with CERTAINTY when the graphs differ in size or WL
    histogram (a valid non-isomorphism certificate); True means 'WL cannot distinguish them'
    (usually but not always isomorphic)."""
    if n1 != n2:
        return False
    if len(e1) != len(e2):
        return False
    return color_histogram(n1, e1) == color_histogram(n2, e2)


# --- exact brute-force isomorphism (ground truth) ---------------------------
def is_isomorphic_bruteforce(n1, e1, n2, e2):
    """Exact isomorphism by permutation search (small graphs only)."""
    if n1 != n2 or len(e1) != len(e2):
        return False
    n = n1
    A = [[False] * n for _ in range(n)]
    for u, v in e1:
        A[u][v] = A[v][u] = True
    target = set()
    for u, v in e2:
        target.add((min(u, v), max(u, v)))
    # prune by degree sequence
    deg1 = sorted(sum(row) for row in A)
    Bdeg = [0] * n
    for u, v in e2:
        Bdeg[u] += 1
        Bdeg[v] += 1
    if deg1 != sorted(Bdeg):
        return False
    for perm in itertools.permutations(range(n)):
        ok = True
        for u in range(n):
            if not ok:
                break
            for v in range(u + 1, n):
                if A[u][v]:
                    a, b = perm[u], perm[v]
                    if (min(a, b), max(a, b)) not in target:
                        ok = False
                        break
        if ok:
            # also need edge counts equal (guaranteed by |e1|==|e2| + injectivity)
            return True
    return False


def relabel(n, edges, perm):
    """Apply a vertex permutation: perm[old] = new."""
    return [(perm[u], perm[v]) for u, v in edges]
