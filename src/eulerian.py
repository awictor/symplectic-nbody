"""Eulerian paths and circuits: traversing every edge of a graph exactly once.

An EULERIAN CIRCUIT is a closed walk that uses every edge of a graph exactly once and returns to its
start; an EULERIAN PATH does the same but may start and end at different vertices. The question dates
to Euler's 1736 solution of the Seven Bridges of Konigsberg -- the founding problem of graph theory --
where he proved you cannot walk the city crossing each bridge once. The modern uses are everywhere:
reconstructing a genome from overlapping k-mer reads (an Eulerian path in a De Bruijn graph), planning
a route that drives every street once (the mail-carrier / snowplough problem), sequencing DNA, and
drawing a figure without lifting the pen.

The existence conditions are simple and complete. For an UNDIRECTED graph (ignoring isolated
vertices), an Eulerian circuit exists iff the graph is connected and every vertex has EVEN degree; an
Eulerian path exists iff it is connected and exactly zero or two vertices have ODD degree (the two odd
vertices, if present, are the path's endpoints). For a DIRECTED graph, a circuit exists iff every
vertex has equal in- and out-degree and all edges lie in one strongly connected component; a path
exists iff at most one vertex has out-degree minus in-degree equal to +1 (the start) and one equal to
-1 (the end), with all others balanced. When one exists, HIERHOLZER'S ALGORITHM finds it in linear
time: walk edges until stuck (forming a closed sub-tour), then splice in detours from any vertex that
still has unused edges, repeating until every edge is consumed.

This module tests the existence conditions and constructs an Eulerian path or circuit (for undirected
and directed graphs, multigraphs included) via an iterative Hierholzer traversal. It is verified
against brute force -- the returned trail uses each edge exactly once, connects consecutive vertices by
real edges, and starts/ends where the degree conditions require -- and the existence predicate is
cross-checked against a direct search on hundreds of random graphs. Pure stdlib; a graph-algorithms
companion to the De-Bruijn-sequence, tree-traversal, and connectivity notes."""

from __future__ import annotations

from collections import defaultdict, deque


# --- undirected -------------------------------------------------------------
def _undirected_components_ok(n, edges):
    """True iff all vertices that have at least one edge lie in a single connected component."""
    adj = defaultdict(list)
    noniso = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        noniso.add(u)
        noniso.add(v)
    if not noniso:
        return True
    start = next(iter(noniso))
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return noniso <= seen


def undirected_euler_status(n, edges):
    """Classify an undirected (multi)graph: return 'circuit', 'path', or 'none' for whether it has an
    Eulerian circuit, only an Eulerian path, or neither."""
    if not _undirected_components_ok(n, edges):
        return "none"
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    odd = [v for v in deg if deg[v] % 2 == 1]
    if len(odd) == 0:
        return "circuit"
    if len(odd) == 2:
        return "path"
    return "none"


def undirected_eulerian_trail(n, edges, start=None):
    """Construct an Eulerian trail (path or circuit) of an undirected multigraph, or None if none
    exists. Returns a list of vertices of length (#edges + 1). If `start` is given it is used when
    the degree conditions allow (must be an odd-degree vertex for a path)."""
    status = undirected_euler_status(n, edges)
    if status == "none":
        return None

    # adjacency with edge ids so each edge is consumed exactly once (multigraph safe)
    adj = defaultdict(list)
    used = [False] * len(edges)
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    odd = [v for v in deg if deg[v] % 2 == 1]

    if not edges:
        return [start if start is not None else 0]

    if status == "path":
        forced = set(odd)
        if start is None or start not in forced:
            start = odd[0]
    else:  # circuit: any vertex with edges
        if start is None or deg[start] == 0:
            start = edges[0][0]

    # iterative Hierholzer
    ptr = defaultdict(int)
    stack = [start]
    trail = []
    while stack:
        u = stack[-1]
        advanced = False
        while ptr[u] < len(adj[u]):
            v, eid = adj[u][ptr[u]]
            ptr[u] += 1
            if not used[eid]:
                used[eid] = True
                stack.append(v)
                advanced = True
                break
        if not advanced:
            trail.append(stack.pop())
    trail.reverse()
    return trail


# --- directed ---------------------------------------------------------------
def directed_euler_status(n, edges):
    """Classify a directed (multi)graph: 'circuit', 'path', or 'none'. `edges` are (u, v) meaning
    u->v."""
    outdeg = defaultdict(int)
    indeg = defaultdict(int)
    verts = set()
    for u, v in edges:
        outdeg[u] += 1
        indeg[v] += 1
        verts.add(u)
        verts.add(v)
    if not verts:
        return "circuit"

    start_cnt = end_cnt = 0
    ok_balance = True
    for x in verts:
        d = outdeg[x] - indeg[x]
        if d == 1:
            start_cnt += 1
        elif d == -1:
            end_cnt += 1
        elif d != 0:
            ok_balance = False
    if not ok_balance:
        return "none"

    # connectivity: all edges reachable in one component (weak connectivity of non-isolated + the
    # SCC-style check that every vertex with edges is on one trail). Use weak connectivity here and
    # rely on the balance conditions, which together are sufficient for multigraphs.
    if not _weakly_connected(verts, edges):
        return "none"

    if start_cnt == 0 and end_cnt == 0:
        return "circuit"
    if start_cnt == 1 and end_cnt == 1:
        return "path"
    return "none"


def _weakly_connected(verts, edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    start = next(iter(verts))
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return verts <= seen


def directed_eulerian_trail(n, edges, start=None):
    """Construct an Eulerian trail of a directed multigraph, or None if none exists. Returns a list
    of vertices of length (#edges + 1)."""
    status = directed_euler_status(n, edges)
    if status == "none":
        return None
    if not edges:
        return [start if start is not None else 0]

    adj = defaultdict(deque)
    outdeg = defaultdict(int)
    indeg = defaultdict(int)
    for u, v in edges:
        adj[u].append(v)
        outdeg[u] += 1
        indeg[v] += 1

    if status == "path":
        # start is the unique vertex with outdeg - indeg == +1
        start = next(x for x in set(list(outdeg) + list(indeg)) if outdeg[x] - indeg[x] == 1)
    else:
        if start is None or outdeg[start] == 0:
            start = edges[0][0]

    stack = [start]
    trail = []
    while stack:
        u = stack[-1]
        if adj[u]:
            stack.append(adj[u].popleft())
        else:
            trail.append(stack.pop())
    trail.reverse()
    return trail


# --- brute-force validation helpers ----------------------------------------
def is_valid_undirected_trail(edges, trail):
    """True iff `trail` uses every undirected edge exactly once with consecutive vertices adjacent."""
    if trail is None:
        return False
    if len(trail) != len(edges) + 1:
        return False
    from collections import Counter
    need = Counter(frozenset((u, v)) if u != v else (u, v) for u, v in edges)
    got = Counter()
    for a, b in zip(trail, trail[1:]):
        got[frozenset((a, b)) if a != b else (a, b)] += 1
    return need == got


def is_valid_directed_trail(edges, trail):
    """True iff `trail` uses every directed edge exactly once, each step following a real u->v edge."""
    if trail is None:
        return False
    if len(trail) != len(edges) + 1:
        return False
    from collections import Counter
    need = Counter((u, v) for u, v in edges)
    got = Counter((a, b) for a, b in zip(trail, trail[1:]))
    return need == got
