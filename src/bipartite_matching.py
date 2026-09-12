"""Maximum bipartite matching: pairing two sides so that as many pairs as possible are made.

A bipartite graph splits its vertices into two disjoint sides -- workers and jobs, students and
projects, medical residents and hospitals -- with edges only between the sides, never within one. A
MATCHING is a set of edges no two of which share a vertex; the MAXIMUM matching pairs up as many as
possible. This is the combinatorial core of assignment: how many workers can be given a job they are
qualified for, how many applicants placed, how many taxis dispatched to riders they can reach. Unlike
the Hungarian algorithm (which minimises total COST on a weighted complete graph), this is the
unweighted question -- maximise the COUNT of pairs -- on an arbitrary bipartite graph where only some
pairs are allowed.

The engine is the AUGMENTING PATH. Given a partial matching, an augmenting path alternates
unmatched/matched edges from a free left vertex to a free right vertex; flipping every edge along it
raises the matching size by exactly one. Berge's theorem says a matching is maximum iff no augmenting
path exists, so repeatedly finding one and flipping it reaches the optimum. HOPCROFT-KARP does this
efficiently: each phase runs a BFS to find the shortest-augmenting-path length, then a DFS to greedily
pack a maximal set of vertex-disjoint shortest paths and flip them all at once. Because the shortest
augmenting-path length strictly increases each phase, only O(sqrt(V)) phases are needed, giving
O(E*sqrt(V)) overall -- far better than the O(V*E) of flipping one path at a time.

The result connects to two classics. KONIG'S THEOREM: in a bipartite graph the size of a maximum
matching equals the size of a minimum vertex cover (the fewest vertices touching every edge), and the
cover is recovered from the alternating-reachability of the final matching. HALL'S THEOREM: a perfect
matching of the left side exists iff every subset S of the left has at least |S| neighbours. This
module computes the maximum matching (Hopcroft-Karp), the minimum vertex cover (via Konig), and the
maximum independent set (its complement). It is verified against brute force -- an exponential search
over all matchings confirms the size is truly maximum on hundreds of random graphs -- and against
Konig's and Hall's theorems directly. Pure stdlib; an optimisation companion to the Hungarian
assignment, max-flow, and Stoer-Wagner min-cut notes."""

from __future__ import annotations

from collections import deque


class BipartiteGraph:
    """Bipartite graph with left vertices 0..nl-1 and right vertices 0..nr-1. Edges connect a left
    vertex to a right vertex."""

    def __init__(self, nl, nr):
        self.nl = nl
        self.nr = nr
        self.adj = [[] for _ in range(nl)]   # adj[u] = list of right-vertices reachable from left u

    def add_edge(self, u, v):
        """Allow pairing left vertex u with right vertex v."""
        self.adj[u].append(v)


_INF = float("inf")


def hopcroft_karp(g):
    """Maximum matching of a bipartite graph, O(E*sqrt(V)).

    Returns (size, match_l, match_r): match_l[u] is the right vertex matched to left u (or -1),
    match_r[v] is the left vertex matched to right v (or -1)."""
    match_l = [-1] * g.nl
    match_r = [-1] * g.nr
    dist = [0] * g.nl

    def bfs():
        """Layer the free-left vertices; return True if any augmenting path exists."""
        q = deque()
        for u in range(g.nl):
            if match_l[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = _INF
        found = False
        while q:
            u = q.popleft()
            for v in g.adj[u]:
                w = match_r[v]                  # left vertex currently matched to v (or -1)
                if w == -1:
                    found = True                # reached a free right vertex -> augmenting path
                elif dist[w] == _INF:
                    dist[w] = dist[u] + 1
                    q.append(w)
        return found

    def dfs(u):
        """Try to find a vertex-disjoint shortest augmenting path from u; flip it if found."""
        for v in g.adj[u]:
            w = match_r[v]
            if w == -1 or (dist[w] == dist[u] + 1 and dfs(w)):
                match_l[u] = v
                match_r[v] = u
                return True
        dist[u] = _INF                          # dead end: don't revisit this phase
        return False

    size = 0
    while bfs():
        for u in range(g.nl):
            if match_l[u] == -1 and dfs(u):
                size += 1
    return size, match_l, match_r


def minimum_vertex_cover(g):
    """Minimum vertex cover via Konig's theorem: from a maximum matching, take the left vertices NOT
    reachable by an alternating path from a free left vertex, plus the right vertices that ARE.

    Returns (left_cover, right_cover) as sorted lists. By Konig, |cover| == max matching size."""
    size, match_l, match_r = hopcroft_karp(g)

    # alternating BFS from every unmatched left vertex, using unmatched edges L->R and matched R->L
    visited_l = [False] * g.nl
    visited_r = [False] * g.nr
    q = deque()
    for u in range(g.nl):
        if match_l[u] == -1:
            visited_l[u] = True
            q.append(u)
    while q:
        u = q.popleft()
        for v in g.adj[u]:
            if v == match_l[u]:                 # skip the matched edge on the way out
                continue
            if not visited_r[v]:
                visited_r[v] = True
                w = match_r[v]
                if w != -1 and not visited_l[w]:
                    visited_l[w] = True
                    q.append(w)

    # Konig cover: unvisited left + visited right
    left_cover = [u for u in range(g.nl) if not visited_l[u]]
    right_cover = [v for v in range(g.nr) if visited_r[v]]
    return left_cover, right_cover


def maximum_independent_set(g):
    """Maximum independent set of a bipartite graph = all vertices minus a minimum vertex cover.
    Returns (left_set, right_set). By Konig-Egervary its size is (nl + nr) - max_matching."""
    left_cover, right_cover = minimum_vertex_cover(g)
    lc, rc = set(left_cover), set(right_cover)
    left_set = [u for u in range(g.nl) if u not in lc]
    right_set = [v for v in range(g.nr) if v not in rc]
    return left_set, right_set


def has_perfect_left_matching(g):
    """True iff every left vertex can be matched (a left-perfect matching exists). By Hall's theorem
    this holds iff every subset S of the left has |neighbours(S)| >= |S|."""
    size, _, _ = hopcroft_karp(g)
    return size == g.nl


# --- brute-force reference --------------------------------------------------
def brute_max_matching(g):
    """Maximum matching size by exhaustive backtracking over left vertices. Exponential; for tests
    on small graphs only."""
    best = [0]

    def rec(u, used_r, count):
        if u == g.nl:
            if count > best[0]:
                best[0] = count
            return
        # option: skip u
        # (prune) even matching all remaining can't beat best
        if count + (g.nl - u) <= best[0]:
            return
        rec(u + 1, used_r, count)
        # option: match u to some free right neighbour
        for v in g.adj[u]:
            if v not in used_r:
                used_r.add(v)
                rec(u + 1, used_r, count + 1)
                used_r.discard(v)

    rec(0, set(), 0)
    return best[0]


def hall_condition_holds(g):
    """Directly test Hall's condition by checking all 2^nl subsets of the left. Exponential; test
    aid only."""
    from itertools import combinations
    for k in range(1, g.nl + 1):
        for subset in combinations(range(g.nl), k):
            neigh = set()
            for u in subset:
                neigh.update(g.adj[u])
            if len(neigh) < len(subset):
                return False
    return True
