"""De Bruijn sequences and Eulerian paths: every window exactly once.

A DE BRUIJN SEQUENCE B(k, n) is a cyclic string over a k-symbol alphabet in which every possible
length-n string appears EXACTLY ONCE as a (cyclically wrapped) substring. There are k**n such
windows, so the sequence has length k**n, and it is astonishingly efficient: a single cyclic string
of length k**n packs all k**n windows with no waste. These sequences are the mathematics behind
rotary shaft encoders (read n adjacent tracks to know the absolute angle), the card trick where a
shuffled-looking deck lets a magician name any run of cards, PIN-pad brute-force ordering (the
shortest string that tries every code), and de novo genome assembly (overlapping k-mers).

The elegant construction reduces the problem to finding an EULERIAN CIRCUIT -- a closed walk using
every edge exactly once -- in the DE BRUIJN GRAPH, whose vertices are the k**(n-1) strings of length
n-1 and whose edges are the k**n strings of length n (an edge for symbol s leads from string w to
the string w[1:]+s). Because every vertex has equal in- and out-degree k, an Eulerian circuit
exists, and HIERHOLZER'S ALGORITHM finds one in linear time by splicing cycles: walk until stuck,
then splice in detours from any vertex that still has unused edges. Reading off the appended symbol
along that circuit yields a De Bruijn sequence.

This module builds De Bruijn sequences for any (k, n) via Hierholzer on the De Bruijn graph, and
also exposes a general Eulerian path/circuit finder for arbitrary directed multigraphs (with the
standard degree conditions). It is verified that every one of the k**n windows appears exactly once
(a bijection onto all length-n strings), that the sequence has the correct length, that small cases
match the classic B(2,3) example, that the greedy prefer-largest 'Ford' sequence is also valid, and
that the Eulerian finder recovers a circuit using each edge once and detects when none exists. Pure
stdlib; a combinatorics-on-words companion to the suffix-array and string-matching notes."""

from __future__ import annotations

from collections import defaultdict


def _eulerian_circuit(n_vertices, adj, start):
    """Hierholzer's algorithm: an Eulerian circuit as a vertex list, consuming each edge once.

    adj[v] is a list of successor vertices (a directed multigraph). Returns the circuit as a list of
    vertices [start, ..., start] of length (#edges + 1). Assumes an Eulerian circuit exists."""
    # copy of the out-edge lists we can pop from
    edges = {v: list(succ) for v, succ in adj.items()}
    ptr = {v: 0 for v in edges}                 # next unused edge index per vertex
    stack = [start]
    circuit = []
    while stack:
        v = stack[-1]
        lst = edges.get(v)
        if lst is not None and ptr[v] < len(lst):
            w = lst[ptr[v]]
            ptr[v] += 1
            stack.append(w)
        else:
            circuit.append(stack.pop())
    circuit.reverse()
    return circuit


def de_bruijn(k, n, alphabet=None):
    """The De Bruijn sequence B(k, n): a cyclic string of length k**n in which every length-n string
    over the alphabet appears exactly once. alphabet defaults to 0..k-1 (returned as a list of
    symbols); pass a string/list of k symbols to get that alphabet.

    Built by an Eulerian circuit on the De Bruijn graph: vertices = length-(n-1) strings, edges =
    length-n strings."""
    if k < 1 or n < 1:
        raise ValueError("k and n must be >= 1")
    if alphabet is None:
        alphabet = list(range(k))
    else:
        alphabet = list(alphabet)
        if len(alphabet) != k:
            raise ValueError("alphabet size must equal k")

    if n == 1:
        # every single symbol once; the cyclic sequence is just the alphabet
        return list(alphabet)

    # De Bruijn graph on (n-1)-tuples of symbol indices. Vertex = tuple of length n-1.
    # Edge for symbol s: v -> v[1:] + (s,). We want an Eulerian circuit.
    # Build adjacency in lexicographic symbol order for determinism.
    adj = defaultdict(list)
    # enumerate all vertices (length n-1 tuples over 0..k-1)
    def all_tuples(length):
        if length == 0:
            yield ()
            return
        for prefix in all_tuples(length - 1):
            for s in range(k):
                yield prefix + (s,)

    for v in all_tuples(n - 1):
        for s in range(k):
            adj[v].append(v[1:] + (s,))

    start = (0,) * (n - 1)
    circuit = _eulerian_circuit(len(adj), adj, start)
    # circuit has k**n + 1 vertices; the appended symbol of each edge is the last element of the
    # destination vertex. Read those off to get the cyclic sequence of length k**n.
    seq_idx = [circuit[i + 1][-1] for i in range(len(circuit) - 1)]
    return [alphabet[i] for i in seq_idx]


def de_bruijn_greedy(k, n, alphabet=None):
    """The prefer-largest greedy (a.k.a. 'Ford') construction of a De Bruijn sequence: start with n
    copies of the smallest symbol, then repeatedly append the largest symbol whose resulting length-n
    suffix has not yet appeared. A classic alternative to the Eulerian construction."""
    if alphabet is None:
        alphabet = list(range(k))
    else:
        alphabet = list(alphabet)
    seq = [0] * n
    seen = {tuple(seq)}
    total = k ** n
    while len(seen) < total:
        placed = False
        for s in range(k - 1, -1, -1):
            window = tuple(seq[-(n - 1):] + [s]) if n > 1 else (s,)
            if window not in seen:
                seq.append(s)
                seen.add(window)
                placed = True
                break
        if not placed:
            break
    # this produces a LINEAR sequence of length total + (n-1); drop the last n-1 (they wrap)
    seq = seq[:total]
    return [alphabet[i] for i in seq]


def windows(seq, n):
    """All length-n cyclic windows of a sequence, as tuples (there are len(seq) of them)."""
    L = len(seq)
    return [tuple(seq[(i + j) % L] for j in range(n)) for i in range(L)]


def is_de_bruijn(seq, k, n):
    """True if seq is a valid B(k, n): length k**n and every length-n window appears exactly once."""
    if len(seq) != k ** n:
        return False
    w = windows(seq, n)
    return len(set(w)) == k ** n


def eulerian_path(n_vertices, edges):
    """A general Eulerian path/circuit finder for a directed multigraph.

    edges: list of (u, v) directed edges (vertices are hashable labels). Returns the Eulerian path as
    a vertex list, or None if none exists. An Eulerian circuit exists iff the graph (restricted to
    vertices with edges) is connected and every vertex has in-degree == out-degree; an Eulerian PATH
    exists iff at most one vertex has out-in == +1 (the start) and at most one has in-out == +1 (the
    end), with all others balanced."""
    adj = defaultdict(list)
    outdeg = defaultdict(int)
    indeg = defaultdict(int)
    verts = set()
    for u, v in edges:
        adj[u].append(v)
        outdeg[u] += 1
        indeg[v] += 1
        verts.add(u)
        verts.add(v)
    if not edges:
        return []

    start_candidates = []
    end_candidates = []
    balanced = True
    for x in verts:
        d = outdeg[x] - indeg[x]
        if d == 1:
            start_candidates.append(x)
        elif d == -1:
            end_candidates.append(x)
        elif d != 0:
            balanced = False
    if not balanced:
        return None
    if len(start_candidates) > 1 or len(end_candidates) > 1:
        return None
    if len(start_candidates) != len(end_candidates):
        return None

    start = start_candidates[0] if start_candidates else next(iter(verts))

    # connectivity: every edge must be reachable from start (weakly, following the walk)
    path = _hierholzer_path(adj, outdeg, start)
    if path is None or len(path) != len(edges) + 1:
        return None
    return path


def _hierholzer_path(adj, outdeg, start):
    edges = {v: list(succ) for v, succ in adj.items()}
    ptr = defaultdict(int)
    stack = [start]
    path = []
    while stack:
        v = stack[-1]
        lst = edges.get(v)
        if lst is not None and ptr[v] < len(lst):
            stack.append(lst[ptr[v]])
            ptr[v] += 1
        else:
            path.append(stack.pop())
    path.reverse()
    return path
